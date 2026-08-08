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
        style = Paint.Style.STROKE; strokeWidth = width*density; strokeCap = Paint.Cap.ROUND; this.color = color
    }
    private val aim = stroke(Color.rgb(70,255,110),4f)
    private val obj = stroke(Color.rgb(255,210,55),5f)
    private val after = stroke(Color.rgb(70,210,255),4f).apply { pathEffect = DashPathEffect(floatArrayOf(16f*density,10f*density),0f) }
    private val bounce = stroke(Color.rgb(255,170,60),4f).apply { pathEffect = DashPathEffect(floatArrayOf(14f*density,10f*density),0f) }
    private val ghost = stroke(Color.WHITE,2.5f)
    private val debug = stroke(Color.argb(160,255,255,255),1.25f)
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.WHITE; textSize=13f*density; setShadowLayer(4f*density,1f,1f,Color.BLACK) }

    fun update(r: AnalysisResult) { result = r; postInvalidateOnAnimation() }

    override fun onDraw(canvas: Canvas) {
        val r = result ?: return
        if (r.frameWidth <= 0 || r.frameHeight <= 0) return
        val sx = width.toDouble()/r.frameWidth; val sy = height.toDouble()/r.frameHeight
        fun x(v:Double)=(v*sx).toFloat(); fun y(v:Double)=(v*sy).toFloat()
        r.segments.forEach { s ->
            val p = when(s.kind){ SegmentKind.AIM->aim; SegmentKind.OBJECT->obj; SegmentKind.CUE_AFTER->after; SegmentKind.BOUNCE->bounce }
            canvas.drawLine(x(s.start.x),y(s.start.y),x(s.end.x),y(s.end.y),p)
        }
        if (r.ghostCueCenter != null && r.cueBall != null) canvas.drawCircle(x(r.ghostCueCenter.x),y(r.ghostCueCenter.y),(r.cueBall.radius*((sx+sy)/2)).toFloat(),ghost)
        val prefs = context.getSharedPreferences(MainActivity.PREFS,Context.MODE_PRIVATE)
        if (prefs.getBoolean(MainActivity.KEY_DEBUG,false)) {
            r.balls.forEach { b -> canvas.drawCircle(x(b.center.x),y(b.center.y),(b.radius*((sx+sy)/2)).toFloat(),debug) }
            r.playRect?.let { canvas.drawRect(x(it.left),y(it.top),x(it.right),y(it.bottom),debug) }
        }
        canvas.drawText("OFFLINE • ${r.status} • ${r.confidence}%",12f*density,24f*density,text)
    }
}
