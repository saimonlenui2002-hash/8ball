package com.example.durakassistant.vision

import android.graphics.Rect

/** Coordinates measured from the supplied 720 x 1574 recording. */
object Realme16Profile {
    const val referenceWidth = 720
    const val referenceHeight = 1574

    fun hand(width: Int, height: Int) = rect(width, height, 0, 1040, 720, 1355)
    fun table(width: Int, height: Int) = rect(width, height, 80, 390, 650, 1050)
    fun trump(width: Int, height: Int) = rect(width, height, 0, 520, 125, 820)
    fun deckCounter(width: Int, height: Int) = rect(width, height, 0, 475, 70, 555)
    fun actionCue(width: Int, height: Int) = rect(width, height, 0, 1340, 285, 1465)

    private fun rect(w: Int, h: Int, l: Int, t: Int, r: Int, b: Int): Rect {
        val sx = w.toFloat() / referenceWidth
        val sy = h.toFloat() / referenceHeight
        return Rect((l * sx).toInt(), (t * sy).toInt(), (r * sx).toInt(), (b * sy).toInt())
    }
}
