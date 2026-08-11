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

    private val aim = stroke(Color.rgb(70, 255, 110), 2.4f)
    private val obj = stroke(Color.rgb(255, 210, 55), 2.8f)
    private val after = stroke(Color.rgb(70, 210, 255), 2.2f).apply {
        pathEffect = DashPathEffect(floatArrayOf(13f * density, 9f * density), 0f)
    }
    private val bounce = stroke(Color.rgb(255, 170, 60), 2.2f).apply {
        pathEffect = DashPathEffect(floatArrayOf(12f * density, 9f * density), 0f)
    }
    // All in-table debug shapes are deliberately saturated. The 6.0 white-guide
    // detector rejects them, so MediaProjection cannot feed our own diagnostics
    // back into the next analysis frame.
    private val ghost = stroke(Color.argb(235, 255, 85, 115), 1.6f)
    private val debug = stroke(Color.argb(180, 180, 80, 255), 1.1f)
    private val cueDebug = stroke(Color.argb(235, 65, 215, 255), 1.7f)
    private val targetDebug = stroke(Color.argb(235, 255, 90, 205), 1.7f)
    private val ghostCross = stroke(Color.argb(245, 255, 85, 115), 1.4f)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 12f * density
        setShadowLayer(4f * density, 1f, 1f, Color.BLACK)
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
        val avgScale = (sx + sy) * 0.5
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
        if (!debugEnabled) return

        r.playRect?.let {
            canvas.drawRect(x(it.left), y(it.top), x(it.right), y(it.bottom), debug)
        }

        val cue = r.cueBall
        val target = r.targetBall
        if (cue != null) {
            canvas.drawCircle(
                x(cue.center.x), y(cue.center.y),
                (cue.radius * avgScale).toFloat(), cueDebug
            )
        }
        if (target != null) {
            canvas.drawCircle(
                x(target.center.x), y(target.center.y),
                (target.radius * avgScale).toFloat(), targetDebug
            )
        }

        r.balls.forEach { b ->
            val sameCue = cue != null && (b.center - cue.center).length() < 1.0
            val sameTarget = target != null && (b.center - target.center).length() < 1.0
            if (!sameCue && !sameTarget) {
                canvas.drawCircle(
                    x(b.center.x), y(b.center.y),
                    (b.radius * avgScale).toFloat(),
                    if (b.isCue) cueDebug else debug
                )
            }
        }

        if (r.ghostCueCenter != null) {
            val g = r.ghostCueCenter
            val rr = ((cue?.radius ?: target?.radius ?: 12.0) * avgScale).toFloat()
            canvas.drawCircle(x(g.x), y(g.y), rr, ghost)
            val cross = 7f * density
            canvas.drawLine(x(g.x) - cross, y(g.y), x(g.x) + cross, y(g.y), ghostCross)
            canvas.drawLine(x(g.x), y(g.y) - cross, x(g.x), y(g.y) + cross, ghostCross)
        }

        if (cue != null && r.aimDirection != null) {
            val d = r.aimDirection.normalized()
            val length = 72.0
            canvas.drawLine(
                x(cue.center.x), y(cue.center.y),
                x(cue.center.x + d.x * length), y(cue.center.y + d.y * length),
                aim
            )
        }

        val cueFlag = if (cue != null) "cue✓" else "cue×"
        val targetFlag = if (target != null) "target✓" else "target×"
        val ghostFlag = if (r.ghostCueCenter != null) "ghost✓" else "ghost×"
        canvas.drawText(
            "DEBUG 6.0 • ${r.status} • ${r.confidence}% • $cueFlag $targetFlag $ghostFlag",
            12f * density,
            24f * density,
            text
        )
    }
}
