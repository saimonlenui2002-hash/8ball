package com.example.pooltrajectory

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.view.View

class TrajectoryOverlayView(context: Context) : View(context) {
    @Volatile private var result: AnalysisResult? = null
    private val density = resources.displayMetrics.density

    private fun stroke(color: Int, width: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = width * density
        strokeCap = Paint.Cap.ROUND
        this.color = color
    }

    private val aim = stroke(Color.rgb(70,255,110),2.4f)
    private val obj = stroke(Color.rgb(255,210,55),2.8f)
    private val after = stroke(Color.rgb(70,210,255),2.2f).apply {
        pathEffect = DashPathEffect(floatArrayOf(13f*density,9f*density),0f)
    }
    private val bounce = stroke(Color.rgb(255,170,60),2.2f).apply {
        pathEffect = DashPathEffect(floatArrayOf(12f*density,9f*density),0f)
    }
    private val ghost = stroke(Color.argb(210,255,255,255),1.5f)
    private val debug = stroke(Color.argb(150,255,255,255),1.0f)
    private val cueDebug = stroke(Color.argb(220,80,220,255),1.6f)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 12f * density
        setShadowLayer(4f*density,1f,1f,Color.BLACK)
    }

    fun update(r: AnalysisResult) {
        result = r
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        val r = result ?: return
        if (r.frameWidth <= 0 || r.frameHeight <= 0) return

        val sx = width.toDouble() / r.frameWidth
        val sy = height.toDouble() / r.frameHeight
        fun x(v: Double) = (v * sx).toFloat()
        fun y(v: Double) = (v * sy).toFloat()

        r.segments.forEach { s ->
            val p = when (s.kind) {
                SegmentKind.AIM -> aim
                SegmentKind.OBJECT -> obj
                SegmentKind.CUE_AFTER -> after
                SegmentKind.BOUNCE -> bounce
            }
            canvas.drawLine(x(s.start.x), y(s.start.y), x(s.end.x), y(s.end.y), p)
        }

        val prefs = context.getSharedPreferences(MainActivity.PREFS, Context.MODE_PRIVATE)
        val debugEnabled = prefs.getBoolean(MainActivity.KEY_DEBUG, false)
        if (debugEnabled) {
            if (r.ghostCueCenter != null && r.cueBall != null) {
                canvas.drawCircle(
                    x(r.ghostCueCenter.x),
                    y(r.ghostCueCenter.y),
                    (r.cueBall.radius * ((sx + sy) / 2)).toFloat(),
                    ghost
                )
            }
            r.balls.forEach { b ->
                canvas.drawCircle(
                    x(b.center.x),
                    y(b.center.y),
                    (b.radius * ((sx + sy) / 2)).toFloat(),
                    if (b.isCue) cueDebug else debug
                )
            }
            r.playRect?.let {
                canvas.drawRect(x(it.left), y(it.top), x(it.right), y(it.bottom), debug)
            }
            canvas.drawText(
                "DEBUG • ${r.status} • ${r.confidence}% • balls=${r.balls.size}",
                12f * density,
                24f * density,
                text
            )
        }
    }
}
