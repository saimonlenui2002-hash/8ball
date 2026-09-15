package com.example.durakassistant.overlay

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import com.example.durakassistant.game.Advice
import com.example.durakassistant.game.GameKnowledge
import kotlin.math.roundToInt

class OverlayController(private val context: Context) {
    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var root: LinearLayout? = null
    private var params: WindowManager.LayoutParams? = null
    private var title: TextView? = null
    private var details: TextView? = null
    private var collapsed = false

    fun show() {
        if (root != null || !Settings.canDrawOverlays(context)) return
        val density = context.resources.displayMetrics.density
        val panel = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            val p = (10 * density).roundToInt()
            setPadding(p, p, p, p)
            background = GradientDrawable().apply {
                cornerRadius = 14 * density
                setColor(Color.argb(222, 20, 40, 49))
                setStroke((1 * density).roundToInt(), Color.rgb(128, 197, 28))
            }
        }
        title = TextView(context).apply {
            setTextColor(Color.rgb(190, 239, 107))
            textSize = 17f
            text = "Анализ…"
        }
        details = TextView(context).apply {
            setTextColor(Color.WHITE)
            textSize = 12.5f
            text = "Ожидаю кадр игры"
        }
        panel.addView(title)
        panel.addView(details)
        panel.setOnClickListener {
            collapsed = !collapsed
            details?.visibility = if (collapsed) View.GONE else View.VISIBLE
        }

        val layout = WindowManager.LayoutParams(
            (218 * density).roundToInt(),
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_SECURE,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (8 * density).roundToInt()
            // Keep the panel above the table recognition area. FLAG_SECURE also
            // prevents MediaProjection from reading the panel's own card text.
            y = (8 * density).roundToInt()
        }
        makeDraggable(panel, layout)
        windowManager.addView(panel, layout)
        root = panel
        params = layout
    }

    fun update(state: GameKnowledge, advice: Advice) {
        if (root == null) show()
        title?.text = advice.title
        val known = state.knownOpponent.sortedBy { it.rank.strength }.joinToString(" ").ifBlank { "—" }
        val possible = state.possibleOpponent
            .filterNot { it in state.knownOpponent }
            .sortedWith(compareBy({ it.suit.ordinal }, { it.rank.strength }))
            .joinToString(" ")
            .let { if (it.length > 72) it.take(69) + "…" else it }
        details?.text = buildString {
            append(advice.detail)
            append("\nСоперник: ${state.opponentCount}")
            append("  Колода: ${state.deckCount}")
            append("\nИзвестно: $known")
            append("\nВозможно: ${possible.ifBlank { "—" }}")
            append("\nРаспознано: ${(state.confidence * 100).roundToInt()}%")
        }
    }

    fun remove() {
        root?.let { runCatching { windowManager.removeView(it) } }
        root = null
        params = null
    }

    private fun makeDraggable(view: View, layout: WindowManager.LayoutParams) {
        var startX = 0
        var startY = 0
        var touchX = 0f
        var touchY = 0f
        var moved = false
        view.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    startX = layout.x; startY = layout.y
                    touchX = event.rawX; touchY = event.rawY
                    moved = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - touchX).roundToInt()
                    val dy = (event.rawY - touchY).roundToInt()
                    moved = moved || kotlin.math.abs(dx) > 8 || kotlin.math.abs(dy) > 8
                    layout.x = startX + dx; layout.y = startY + dy
                    windowManager.updateViewLayout(view, layout)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!moved) view.performClick()
                    true
                }
                else -> false
            }
        }
    }
}
