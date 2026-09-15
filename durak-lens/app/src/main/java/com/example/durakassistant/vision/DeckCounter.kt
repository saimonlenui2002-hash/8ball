package com.example.durakassistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
import com.example.durakassistant.R
import kotlin.math.abs

class DeckCounter(context: Context) {
    private val templates = mapOf(
        12 to load(context, R.drawable.deck_12),
        10 to load(context, R.drawable.deck_10),
        8 to load(context, R.drawable.deck_8),
        6 to load(context, R.drawable.deck_6),
        4 to load(context, R.drawable.deck_4),
        2 to load(context, R.drawable.deck_2)
    )

    fun read(bitmap: Bitmap, region: Rect): Pair<Int, Float>? {
        val sample = mask(bitmap, region)
        val ink = sample.count { it }
        if (ink < 8) return 0 to 0.9f
        val best = templates.map { (count, template) -> count to similarity(sample, template) }.maxByOrNull { it.second }
            ?: return null
        return best.takeIf { it.second >= 0.42f }
    }

    private fun load(context: Context, id: Int): BooleanArray {
        val bitmap = BitmapFactory.decodeResource(context.resources, id)
        return mask(bitmap, Rect(0, 0, bitmap.width, bitmap.height))
    }

    private fun mask(bitmap: Bitmap, region: Rect): BooleanArray {
        val out = BooleanArray(W * H)
        for (y in 0 until H) for (x in 0 until W) {
            val sx = region.left + x * region.width() / W
            val sy = region.top + y * region.height() / H
            val c = bitmap.getPixel(sx.coerceIn(0, bitmap.width - 1), sy.coerceIn(0, bitmap.height - 1))
            val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
            out[y * W + x] = r > 185 && g > 185 && b > 185 && abs(r - g) < 30
        }
        return out
    }

    private fun similarity(a: BooleanArray, b: BooleanArray): Float {
        var overlap = 0
        var total = 0
        for (i in a.indices) {
            if (a[i]) total++
            if (b[i]) total++
            if (a[i] && b[i]) overlap++
        }
        return if (total == 0) 0f else (2f * overlap) / total
    }

    companion object { private const val W = 35; private const val H = 40 }
}
