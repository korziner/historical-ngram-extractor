//! main.rs
//! 
//! Зависимости (Cargo.toml):
//! [dependencies]
//! clap = { version = "4.4", features = ["derive", "env"] }
//! indicatif = "0.17"
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! rust-stemmers = "1.2"
//! tantivy = "0.22"
//! rayon = "1.8"
//! rustc-hash = "2.0"
//! sbbf-rs-safe = "0.2"
//! wyhash = "0.5"
//!
//! Использование:
//! RAYON_NUM_THREADS=16 zstdcat corpus.zst | historical-ngram-extractor -n 7 -f 9 --dict dict.txt > out.tsv

use std::io::{self, BufRead, BufWriter, Write};
use std::fs::File;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::borrow::Cow;
use std::sync::Arc;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use rust_stemmers::{Algorithm, Stemmer};
use tantivy::tokenizer::{SimpleTokenizer, TokenStream, Tokenizer};
use rayon::prelude::*;
use rustc_hash::FxHasher;
use sbbf_rs_safe::Filter;
use wyhash::wyhash;

type FastStringCounter = HashMap<String, u64, BuildHasherDefault<FxHasher>>;
type FastStringSet = HashMap<String, (), BuildHasherDefault<FxHasher>>;

/// Оптимизированный индекс словаря для параллельного доступа
struct LexicalDictionaryIndex {
    exact_set: Arc<FastStringSet>,
    bloom_filter: Arc<Filter>,
    word_hashes: Arc<Vec<u64>>,
    hasher_seed: u64,
}

#[derive(Parser, Debug)]
#[command(name = "historical-ngram-extractor", version, about = "Экстрактор уникальных n-грамм с поддержкой дореформенной орфографии", long_about = None)]
struct CliArgs {
    #[arg(short = 'n', long, default_value_t = 2, env = "N")]
    ngram_size: usize,
    #[arg(short = 'f', long = "min-freq", default_value_t = 3, env = "MIN_FREQ")]
    min_freq: u64,
    #[arg(short, long, env = "DICT_PATH")]
    dict_path: Option<String>,
    #[arg(short, long, default_value = "ru", env = "STEMMER_LANG")]
    language: String,
    #[arg(long)]
    noprogress: bool,
    /// Число потоков для rayon (переопределяет RAYON_NUM_THREADS)
    #[arg(long, env = "RAYON_THREADS")]
    threads: Option<usize>,
}

#[derive(Deserialize)]
struct JsonlRecord {
    text: Option<String>,
    #[allow(dead_code)] date: Option<String>,
    #[allow(dead_code)] title: Option<String>,
}

#[inline] fn is_cyrillic_character(c: char) -> bool {
    matches!(c, '\u{0400}'..='\u{04FF}' | '\u{0500}'..='\u{052F}')
}

#[inline] fn passes_ocr_heuristics(token: &str) -> bool {
    if token.len() < 2 { return false; }
    let (mut vowels, mut consec, mut prev) = (0, 0, '\0');
    for c in token.chars() {
        if !is_cyrillic_character(c) && c != '-' && c != '\'' { return false; }
        if "аеёиоуыэюяАЕЁИОУЫЭЮЯ".contains(c) { vowels += 1; }
        if c == prev { consec += 1; if consec >= 3 { return false; } } else { consec = 0; }
        prev = c;
    }
    vowels > 0
}

#[inline] fn normalize_to_modern_orthography(token: &str) -> Cow<str> {
    if !token.contains(['ѣ', 'і', 'ѳ', 'ѵ', 'ъ']) { return Cow::Borrowed(token); }
    let mut out = String::with_capacity(token.len());
    for c in token.chars() {
        match c {
            'ѣ' => out.push('е'), 'і' => out.push('и'), 'ѳ' => out.push('ф'),
            'ѵ' => out.push('и'), 'ъ' => continue, _ => out.push(c),
        }
    }
    Cow::Owned(out)
}

#[inline] fn convert_to_pre_reform_orthography(token: &str) -> Cow<str> {
    let mut needs = token.contains(['ё', 'э']);
    if let Some(last) = token.chars().last() {
        if matches!(last, 'б'|'в'|'г'|'д'|'ж'|'з'|'к'|'л'|'м'|'н'|'п'|'р'|'с'|'т'|'ф'|'х'|'ц'|'ч'|'ш'|'щ') { needs = true; }
    }
    if !needs { return Cow::Borrowed(token); }
    let mut out = String::with_capacity(token.len() + 1);
    for c in token.chars() { if matches!(c, 'ё'|'э') { out.push('е'); } else { out.push(c); } }
    if let Some(last) = out.chars().last() {
        if matches!(last, 'б'|'в'|'г'|'д'|'ж'|'з'|'к'|'л'|'м'|'н'|'п'|'р'|'с'|'т'|'ф'|'х'|'ц'|'ч'|'ш'|'щ') { out.push('ъ'); }
    }
    Cow::Owned(out)
}

#[inline] fn hash_string_for_bloom(text: &str, seed: u64) -> u64 {
    wyhash(text.as_bytes(), seed)
}

/// Параллельная загрузка словаря с построением индекса
fn load_lexical_dictionary_parallel(dictionary_file_path: &str, seed: u64) -> LexicalDictionaryIndex {
    let file = File::open(dictionary_file_path).expect("Не удалось открыть словарь");
    let reader = io::BufReader::new(file);
    
    // Параллельная нормализация и хеширование слов
    let words: Vec<String> = reader.lines()
        .filter_map(|r| r.ok())
        .map(|l| l.trim().to_lowercase())
        .map(|w| normalize_to_modern_orthography(&w).into_owned())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .collect();
    
    let word_count = words.len();
    let bloom_capacity = (word_count as f64 * 1.2) as usize;
    
    // Параллельное вычисление хешей
    let word_hashes: Vec<u64> = words.par_iter()
        .map(|w| hash_string_for_bloom(w, seed))
        .collect();
    
    // Построение exact_set (параллельная вставка в FxHashMap требует осторожности — используем collect)
    let exact_set: FastStringSet = words.into_par_iter()
        .map(|w| (w, ()))
        .collect();
    
    // Построение Bloom-фильтра (однопоточно, но быстро)
    let mut bloom_filter = Filter::new(12, bloom_capacity.max(1024));
    for &hash in &word_hashes {
        bloom_filter.insert_hash(hash);
    }
    
    LexicalDictionaryIndex {
        exact_set: Arc::new(exact_set),
        bloom_filter: Arc::new(bloom_filter),
        word_hashes: Arc::new(word_hashes),
        hasher_seed: seed,
    }
}

fn extract_ngrams_from_document(
    document_text: &str,
    ngram_length: usize,
    word_stemmer: &Stemmer,
    text_tokenizer: &SimpleTokenizer,
    dict_index: Option<&LexicalDictionaryIndex>,
) -> Vec<String> {
    let mut stems = Vec::with_capacity(1024);
    let mut local_tokenizer = text_tokenizer.clone();
    let mut token_stream = local_tokenizer.token_stream(document_text);
    
    while token_stream.advance() {
        let raw = token_stream.token().text.as_str();
        let modern = normalize_to_modern_orthography(raw);
        
        let in_dict = dict_index.map_or(false, |idx| {
            let h = hash_string_for_bloom(&modern, idx.hasher_seed);
            // Быстрая проверка: если хеша нет в предвычисленном списке — пропускаем точную проверку
            // Это снижает нагрузку на кэш при работе с большими словарями
            if !idx.word_hashes.binary_search(&h).is_ok() && !idx.bloom_filter.contains_hash(h) {
                return false;
            }
            idx.exact_set.contains_key(modern.as_ref())
        });
        
        if !in_dict && !passes_ocr_heuristics(&modern) { continue; }
        
        let stemmed = word_stemmer.stem(&modern);
        let pre = convert_to_pre_reform_orthography(&stemmed);
        if pre.len() > 1 { stems.push(pre.into_owned()); }
    }
    
    if stems.len() < ngram_length { return vec![]; }
    
    let mut ngrams = Vec::with_capacity(stems.len() - ngram_length + 1);
    for window in stems.windows(ngram_length) {
        let cap = window.iter().map(|s| s.len()).sum::<usize>() + (ngram_length - 1);
        let mut ng = String::with_capacity(cap);
        for (i, w) in window.iter().enumerate() { if i > 0 { ng.push(' '); } ng.push_str(w); }
        ngrams.push(ng);
    }
    ngrams
}

fn compute_batch_ngram_frequencies(
    batch_lines: &[String],
    ngram_length: usize,
    word_stemmer: &Stemmer,
    text_tokenizer: &SimpleTokenizer,
    dict_index: Option<&LexicalDictionaryIndex>,
) -> FastStringCounter {
    batch_lines.par_iter()
        .filter_map(|line| {
            let t = line.trim();
            if t.is_empty() { return None; }
            if let Ok(rec) = serde_json::from_str::<JsonlRecord>(t) {
                rec.text.filter(|x| !x.is_empty())
            } else {
                Some(t.to_string())
            }
        })
        .flat_map(|txt| extract_ngrams_from_document(&txt, ngram_length, word_stemmer, text_tokenizer, dict_index))
        .fold(FastStringCounter::default, |mut acc, ng| { *acc.entry(ng).or_insert(0) += 1; acc })
        .reduce(FastStringCounter::default, |mut a, mut b| {
            if a.len() < b.len() { for (k,v) in a { *b.entry(k).or_insert(0) += v; } b }
            else { for (k,v) in b { *a.entry(k).or_insert(0) += v; } a }
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();
    if args.ngram_size < 2 { eprintln!("Ошибка: n-грамма должна быть >= 2"); std::process::exit(1); }
    
    // Настройка пула потоков rayon ДО любой параллельной операции
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("Не удалось создать пул потоков rayon");
    }
    
    let algo = match args.language.to_lowercase().as_str() {
        "en" => Algorithm::English, "ru" => Algorithm::Russian,
        "de" => Algorithm::German, "fr" => Algorithm::French,
        "es" => Algorithm::Spanish, _ => {
            eprintln!("Предупреждение: неизвестный язык '{}'. Используется русский.", args.language);
            Algorithm::Russian
        }
    };
    
    const HASH_SEED: u64 = 0x9e3779b97f4a7c15;
    let dict_index = args.dict_path.as_ref().map(|p| load_lexical_dictionary_parallel(p, HASH_SEED));
    
    let stemmer = Stemmer::create(algo);
    let tokenizer = SimpleTokenizer::default();
    const BATCH_SIZE: usize = 4096;
    
    eprintln!("[INFO] Старт. N={}, MinFreq={}, Язык={}, Dict={}, Потоки={}", 
              args.ngram_size, args.min_freq, args.language,
              args.dict_path.as_deref().unwrap_or("отключён"),
              rayon::current_num_threads());
    
    let pb = if args.noprogress { ProgressBar::hidden() } else {
        let p = ProgressBar::new_spinner();
        p.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg} ({per_sec})").unwrap()
            .progress_chars("##-"));
        p.set_message("Чтение..."); p
    };
    
    let stdin = io::BufReader::new(io::stdin().lock());
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut total_lines: u64 = 0;
    let mut global_counts = FastStringCounter::default();
    
    for line in stdin.lines() {
        batch.push(line?);
        if batch.len() >= BATCH_SIZE {
            let mut local = compute_batch_ngram_frequencies(&batch, args.ngram_size, &stemmer, &tokenizer, dict_index.as_ref());
            total_lines += batch.len() as u64;
            pb.inc(batch.len() as u64);
            pb.set_message(format!("Строк: {}", total_lines));
            
            if global_counts.len() < local.len() {
                for (k,v) in global_counts { *local.entry(k).or_insert(0) += v; }
                global_counts = local;
            } else {
                for (k,v) in local { *global_counts.entry(k).or_insert(0) += v; }
            }
            batch.clear();
        }
    }
    if !batch.is_empty() {
        let mut local = compute_batch_ngram_frequencies(&batch, args.ngram_size, &stemmer, &tokenizer, dict_index.as_ref());
        total_lines += batch.len() as u64;
        if global_counts.len() < local.len() {
            for (k,v) in global_counts { *local.entry(k).or_insert(0) += v; }
            global_counts = local;
        } else {
            for (k,v) in local { *global_counts.entry(k).or_insert(0) += v; }
        }
    }
    
    pb.finish_with_message("Фильтрация...");
    
    let mut result: Vec<(String, u64)> = global_counts.into_iter()
        .filter(|(_,c)| *c >= args.min_freq)
        .collect();
    result.sort_unstable_by(|a,b| a.0.cmp(&b.0));
    
    eprintln!("[INFO] Завершено. Строк: {}, {}-грамм: {}", total_lines, args.ngram_size, result.len());
    
    let mut out = BufWriter::new(io::stdout().lock());
    for (ng, cnt) in result { writeln!(out, "{}\t{}", ng, cnt)?; }
    out.flush()?;
    
    Ok(())
}
