package com.example.fastvlm.ondevice

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.AttributeSet
import android.view.View

/** One GeoChat box in 0-100 image coordinates; angle in degrees (cv2 convention, counter-clockwise). */
data class GeoBox(val label: String, val x1: Float, val y1: Float, val x2: Float, val y2: Float, val angle: Float)

/**
 * Parses GeoChat answers like `<p>two ships</p> {<78><97><82><99>|<52>}<delim>{...}`.
 * Same result as geochat_demo.py on well-formed answers, but also tolerates a box written inside the
 * phrase (`<p>two airplanes{<23><10><63><46>|<79>} and</p> {...}`), which the model sometimes emits.
 */
object GeoChatFormat {
    const val SCALE = 100f
    private val BOX = Regex("""\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>(?:\|<(-?\d+)>)?\}""")
    private val TOKEN = Regex("""<p>|</p>|""" + BOX.pattern)

    // ARGB literals (not Color.rgb) so this object also loads in plain-JVM unit tests
    val PALETTE = intArrayOf(
        0xFFFF0000.toInt(), 0xFF00C800.toInt(), 0xFF005AFF.toInt(), 0xFFD2D200.toInt(),
        0xFFFF00FF.toInt(), 0xFF00DCDC.toInt(), 0xFFFFA500.toInt(), 0xFF8A2BE2.toInt(),
    )

    /** Each box takes the label of the closest preceding phrase; a box inside an open <p> ends that phrase. */
    fun parseBoxes(text: String): List<GeoBox> {
        val boxes = ArrayList<GeoBox>()
        var label = ""
        var phraseStart = -1   // index after an open <p>, or -1
        for (m in TOKEN.findAll(text)) {
            when (m.value) {
                "<p>" -> phraseStart = m.range.last + 1
                "</p>" -> if (phraseStart >= 0) { label = text.substring(phraseStart, m.range.first).trim(); phraseStart = -1 }
                else -> {
                    if (phraseStart >= 0) { label = text.substring(phraseStart, m.range.first).trim(); phraseStart = -1 }
                    val v = (1..4).map { m.groupValues[it].toFloat() }
                    val angle = m.groups[5]?.value?.toFloat() ?: 0f
                    boxes.add(GeoBox(label, v[0], v[1], v[2], v[3], angle))
                }
            }
        }
        return boxes
    }

    /** Boxes with identical coordinates (the model sometimes repeats one box for two phrases) shown once, labels joined. */
    fun mergeDuplicates(boxes: List<GeoBox>): List<GeoBox> {
        val byCoords = LinkedHashMap<List<Float>, GeoBox>()
        for (b in boxes) {
            val key = listOf(b.x1, b.y1, b.x2, b.y2, b.angle)
            val prev = byCoords[key]
            byCoords[key] = if (prev == null || prev.label == b.label || b.label.isEmpty()) prev ?: b
                            else prev.copy(label = if (prev.label.isEmpty()) b.label else "${prev.label} / ${b.label}")
        }
        return byCoords.values.toList()
    }

    /** Stable color per label, in order of first appearance. */
    fun colorsFor(boxes: List<GeoBox>): Map<String, Int> {
        val colors = LinkedHashMap<String, Int>()
        for (b in boxes) colors.getOrPut(b.label) { PALETTE[colors.size % PALETTE.size] }
        return colors
    }

    /** Readable answer: box strings removed, <p>phrases</p> shown bold in their box color. */
    fun styledAnswer(text: String, colors: Map<String, Int>): CharSequence {
        val clean = text.replace(BOX, "").replace("<delim>", "").replace(Regex("""[ \t]{2,}"""), " ")
        val out = SpannableStringBuilder()
        var last = 0
        for (m in Regex("""<p>(.*?)</p>""").findAll(clean)) {
            out.append(clean, last, m.range.first)
            val start = out.length
            val phrase = m.groupValues[1]
            out.append(phrase)
            // Labels can be merged ("a / b") or cut short by a box inside the phrase, so match loosely
            val color = colors.entries.firstOrNull { (label, _) ->
                label.isNotEmpty() && (label.split(" / ").any { it == phrase.trim() } || phrase.trim().startsWith(label))
            }?.value ?: PALETTE[0]
            out.setSpan(ForegroundColorSpan(color), start, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            out.setSpan(StyleSpan(android.graphics.Typeface.BOLD), start, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            last = m.range.last + 1
        }
        out.append(clean.substring(last).replace("<p>", "").replace("</p>", "").replace(Regex("</?p?$"), ""))
        return out
    }

    fun boxString(x1: Int, y1: Int, x2: Int, y2: Int, angle: Int = 0) = "{<$x1><$y1><$x2><$y2>|<$angle>}"
}

/** Draws GeoChat boxes over the area where the image is shown ([imageRect], in view coordinates). */
class BoxOverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var boxes: List<GeoBox> = emptyList()
    private var colors: Map<String, Int> = emptyMap()
    private var imageRect: RectF? = null

    private val density = resources.displayMetrics.density
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 3 * density }
    private val labelBg = Paint().apply { style = Paint.Style.FILL }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.BLACK; textSize = 12 * density }

    fun show(boxes: List<GeoBox>, colors: Map<String, Int>, imageRect: RectF?) {
        this.boxes = boxes
        this.colors = colors
        this.imageRect = imageRect
        invalidate()
    }

    fun clear() = show(emptyList(), emptyMap(), null)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = imageRect ?: RectF(0f, 0f, width.toFloat(), height.toFloat())
        for (b in boxes) {
            val color = colors[b.label] ?: GeoChatFormat.PALETTE[0]
            stroke.color = color
            val left = r.left + b.x1 / GeoChatFormat.SCALE * r.width()
            val top = r.top + b.y1 / GeoChatFormat.SCALE * r.height()
            val right = r.left + b.x2 / GeoChatFormat.SCALE * r.width()
            val bottom = r.top + b.y2 / GeoChatFormat.SCALE * r.height()
            canvas.save()
            // cv2.getRotationMatrix2D(center, angle) is counter-clockwise on screen; Canvas.rotate is clockwise
            canvas.rotate(-b.angle, (left + right) / 2, (top + bottom) / 2)
            canvas.drawRect(left, top, right, bottom, stroke)
            canvas.restore()
            if (b.label.isNotEmpty()) {
                val tw = labelText.measureText(b.label)
                val th = labelText.textSize
                val ty = maxOf(top, r.top + th + 4 * density)
                labelBg.color = color
                canvas.drawRect(left, ty - th - 4 * density, left + tw + 6 * density, ty, labelBg)
                canvas.drawText(b.label, left + 3 * density, ty - 3 * density, labelText)
            }
        }
    }
}
