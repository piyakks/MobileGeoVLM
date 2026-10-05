package com.example.fastvlm.ondevice

import org.json.JSONObject
import java.io.File
import java.text.Normalizer
import java.util.regex.Pattern

/**
 * Byte-level BPE tokenizer for Qwen2 (vocab.json + merges.txt).
 * Mirrors HF tokenizers: NFC normalize -> regex split -> byte-to-unicode -> BPE merges.
 */
class QwenTokenizer(modelDir: File) {

    private val vocab = HashMap<String, Int>(160_000)
    private val idToToken = HashMap<Int, String>(160_000)
    private val ranks = HashMap<String, Int>(160_000)
    private val cache = HashMap<String, IntArray>()

    private val special = mapOf(
        "<|endoftext|>" to 151643,
        "<|im_start|>" to 151644,
        "<|im_end|>" to 151645,
    )
    private val specialPattern = Pattern.compile(special.keys.joinToString("|") { Pattern.quote(it) })

    // Qwen2 pre-tokenizer regex. \s is spelled out as Unicode White_Space (as in the Rust regex) because
    // Android's regex engine does not support UNICODE_CHARACTER_CLASS.
    private val ws = "\\t\\n\\u000B\\f\\r\\u0085\\p{Z}"
    private val splitPattern = Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^$ws\\p{L}\\p{N}]+[\\r\\n]*" +
            "|[$ws]*[\\r\\n]+|[$ws]+(?![^$ws])|[$ws]+"
    )

    private val byteToUnicode = CharArray(256)
    private val unicodeToByte = HashMap<Char, Int>(256)

    init {
        // GPT-2 byte <-> unicode table
        val bs = ArrayList<Int>()
        bs.addAll('!'.code..'~'.code); bs.addAll('¡'.code..'¬'.code); bs.addAll('®'.code..'ÿ'.code)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0 until 256) if (b !in bs) { bs.add(b); cs.add(256 + n); n++ }
        for (i in bs.indices) {
            byteToUnicode[bs[i]] = cs[i].toChar()
            unicodeToByte[cs[i].toChar()] = bs[i]
        }

        val json = JSONObject(File(modelDir, "vocab.json").readText())
        val keys = json.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val id = json.getInt(k)
            vocab[k] = id
            idToToken[id] = k
        }
        for ((k, v) in special) idToToken[v] = k

        var rank = 0
        File(modelDir, "merges.txt").forEachLine { line ->
            if (line.isEmpty() || line.startsWith("#version")) return@forEachLine
            ranks[line] = rank++   // key is "a b"
        }
    }

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        val m = specialPattern.matcher(text)
        var last = 0
        while (m.find()) {
            if (m.start() > last) encodeOrdinary(text.substring(last, m.start()), out)
            out.add(special.getValue(m.group()))
            last = m.end()
        }
        if (last < text.length) encodeOrdinary(text.substring(last), out)
        return out.toIntArray()
    }

    private fun encodeOrdinary(text: String, out: MutableList<Int>) {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFC)
        val m = splitPattern.matcher(normalized)
        while (m.find()) {
            val piece = m.group()
            val bytes = piece.toByteArray(Charsets.UTF_8)
            val sb = StringBuilder(bytes.size)
            for (b in bytes) sb.append(byteToUnicode[b.toInt() and 0xFF])
            for (id in bpe(sb.toString())) out.add(id)
        }
    }

    private fun bpe(word: String): IntArray {
        cache[word]?.let { return it }
        val parts = ArrayList<String>(word.length)
        for (c in word) parts.add(c.toString())
        while (parts.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until parts.size - 1) {
                val r = ranks[parts[i] + " " + parts[i + 1]] ?: continue
                if (r < bestRank) { bestRank = r; bestIdx = i }
            }
            if (bestIdx < 0) break
            parts[bestIdx] = parts[bestIdx] + parts[bestIdx + 1]
            parts.removeAt(bestIdx + 1)
        }
        val ids = IntArray(parts.size) { vocab.getValue(parts[it]) }
        if (cache.size < 10_000) cache[word] = ids
        return ids
    }

    fun decode(ids: List<Int>): String {
        val bytes = java.io.ByteArrayOutputStream()
        for (id in ids) {
            if (id in special.values) continue
            val token = idToToken[id] ?: continue
            for (c in token) unicodeToByte[c]?.let { bytes.write(it) }
        }
        return bytes.toString(Charsets.UTF_8.name())
    }
}
