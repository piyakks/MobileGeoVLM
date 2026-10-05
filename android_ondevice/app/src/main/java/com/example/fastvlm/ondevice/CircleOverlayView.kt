package com.example.fastvlm.ondevice

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

/**
 * Transparent overlay on top of the camera preview where the user draws one red circle.
 * The stroke is kept in view coordinates and can be burned into a frame that covers the same area.
 */
class CircleOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val strokeDp = 6f
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        strokeWidth = strokeDp * resources.displayMetrics.density
    }

    private val points = ArrayList<Pair<Float, Float>>()

    /** Incremented whenever the circle changes, so callers can tell a new circle from an old one. */
    @Volatile var version = 0
        private set

    var drawingEnabled = false

    private var drawing = false

    /** True once a finished circle exists (not while the finger is still drawing). */
    val hasCircle get() = points.size > 1 && !drawing

    fun clear() {
        points.clear()
        version++
        invalidate()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (!drawingEnabled) return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> { drawing = true; points.clear(); points.add(event.x to event.y) }  // a new circle replaces the old one
            MotionEvent.ACTION_MOVE -> points.add(event.x to event.y)
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { drawing = false; points.add(event.x to event.y); version++ }
        }
        invalidate()
        return true
    }

    private fun buildPath(scaleX: Float, scaleY: Float, offsetX: Float = 0f, offsetY: Float = 0f): Path {
        val path = Path()
        points.forEachIndexed { i, (x, y) ->
            val px = (x - offsetX) * scaleX
            val py = (y - offsetY) * scaleY
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        return path
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (points.size > 1) canvas.drawPath(buildPath(1f, 1f), paint)
    }

    private fun area(imageRect: RectF?) = imageRect ?: RectF(0f, 0f, width.toFloat(), height.toFloat())

    /**
     * Returns a copy of [frame] with the circle drawn on it. [imageRect] is where [frame] is shown in this
     * view (null = the whole view, e.g. the live preview whose frames are cropped to the same viewport).
     */
    fun renderOnto(frame: Bitmap, imageRect: RectF? = null): Bitmap {
        val out = frame.copy(Bitmap.Config.ARGB_8888, true)
        val r = area(imageRect)
        if (!hasCircle || r.width() <= 0f || r.height() <= 0f) return out
        val sx = out.width / r.width()
        val sy = out.height / r.height()
        val p = Paint(paint).apply { strokeWidth = paint.strokeWidth * sx }
        Canvas(out).drawPath(buildPath(sx, sy, r.left, r.top), p)
        return out
    }

    /** Bounding box of the drawing in GeoChat's 0-100 image coordinates (x1, y1, x2, y2), or null. */
    fun boundingBox(imageRect: RectF? = null): IntArray? {
        if (!hasCircle) return null
        val r = area(imageRect)
        if (r.width() <= 0f || r.height() <= 0f) return null
        fun nx(x: Float) = ((x - r.left) / r.width() * 100f).toInt().coerceIn(0, 100)
        fun ny(y: Float) = ((y - r.top) / r.height() * 100f).toInt().coerceIn(0, 100)
        val box = intArrayOf(nx(points.minOf { it.first }), ny(points.minOf { it.second }),
                             nx(points.maxOf { it.first }), ny(points.maxOf { it.second }))
        return if (box[2] > box[0] && box[3] > box[1]) box else null
    }
}
