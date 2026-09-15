package com.example.durakassistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
import com.example.durakassistant.R
import com.example.durakassistant.game.Card
import com.example.durakassistant.game.Rank
import com.example.durakassistant.game.Suit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

data class CardDetection(val card: Card, val bounds: Rect, val confidence: Float)

class GlyphMatcher(context: Context) {
    private val rankTemplates: Map<Rank, Shape>
    private val suitTemplates: Map<Suit, Shape>

    init {
        rankTemplates = mapOf(
            Rank.NINE to shapeFromResource(context, R.drawable.rank_9, false),
            Rank.TEN to shapeFromResource(context, R.drawable.rank_10, false, keepTwo = true),
            Rank.JACK to shapeFromResource(context, R.drawable.rank_v, false),
            Rank.QUEEN to shapeFromResource(context, R.drawable.rank_d, false),
            Rank.KING to shapeFromResource(context, R.drawable.rank_k, false),
            Rank.ACE to shapeFromResource(context, R.drawable.rank_t, false)
        )
        suitTemplates = mapOf(
            Suit.CLUBS to shapeFromResource(context, R.drawable.suit_clubs, true),
            Suit.DIAMONDS to shapeFromResource(context, R.drawable.suit_diamonds, true),
            Suit.HEARTS to shapeFromResource(context, R.drawable.suit_hearts, true),
            Suit.SPADES to shapeFromResource(context, R.drawable.suit_spades, true)
        )
    }

    fun findCards(bitmap: Bitmap, region: Rect): List<CardDetection> {
        val components = components(bitmap, region)
        val rankParts = components + mergeForTen(components)
        val ranks = rankParts.mapNotNull { component ->
            val match = best(component.shape(), rankTemplates) ?: return@mapNotNull null
            if (match.second < 0.49f) return@mapNotNull null
            ClassifiedRank(component, match.first, match.second)
        }
        val suits = components.mapNotNull { component ->
            val match = best(component.shape(), suitTemplates) ?: return@mapNotNull null
            if (match.second < 0.45f) return@mapNotNull null
            ClassifiedSuit(component, match.first, match.second)
        }

        val detections = mutableListOf<CardDetection>()
        for (rank in ranks.sortedByDescending { it.score }) {
            val rb = rank.component.bounds
            val maxGap = max(24, rb.height() * 2)
            val suit = suits
                .asSequence()
                .filter { it.suit.red == rank.component.mostlyRed }
                .filter {
                    val sb = it.component.bounds
                    sb.top >= rb.top + rb.height() / 3 && sb.top <= rb.bottom + maxGap &&
                        abs(sb.centerX() - rb.centerX()) < max(rb.width() * 2, 45)
                }
                .maxByOrNull { it.score - distancePenalty(rb, it.component.bounds) }
                ?: continue

            val combined = Rect(
                min(rb.left, suit.component.bounds.left), rb.top,
                max(rb.right, suit.component.bounds.right), suit.component.bounds.bottom
            )
            val confidence = (rank.score + suit.score) / 2f
            val detection = CardDetection(Card(rank.rank, suit.suit), combined, confidence)
            if (detections.none { sameSpot(it.bounds, combined) }) detections += detection
        }
        return detections.sortedWith(compareBy<CardDetection> { it.bounds.top }.thenBy { it.bounds.left })
    }

    fun findTrumpSuit(bitmap: Bitmap, region: Rect): Pair<Suit, Float>? {
        return components(bitmap, region)
            .flatMap { component -> suitTemplates.map { (suit, shape) -> Triple(suit, similarity(component.shape(), shape), component) } }
            .filter { (suit, _, component) -> suit.red == component.mostlyRed }
            .maxByOrNull { it.second }
            ?.takeIf { it.second >= 0.42f }
            ?.let { it.first to it.second }
    }

    private fun components(bitmap: Bitmap, region: Rect): List<Component> {
        val w = region.width()
        val h = region.height()
        if (w <= 0 || h <= 0) return emptyList()
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, region.left, region.top, w, h)
        val foreground = BooleanArray(pixels.size) { isInk(pixels[it]) }
        val seen = BooleanArray(pixels.size)
        val output = mutableListOf<Component>()
        val queue = IntArray(pixels.size)

        for (start in foreground.indices) {
            if (!foreground[start] || seen[start]) continue
            var head = 0
            var tail = 0
            queue[tail++] = start
            seen[start] = true
            val points = ArrayList<Int>(128)
            var red = 0
            var minX = w
            var minY = h
            var maxX = 0
            var maxY = 0
            while (head < tail) {
                val p = queue[head++]
                points += p
                val x = p % w
                val y = p / w
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                if (isRed(pixels[p])) red++
                if (x > 0) add(p - 1, foreground, seen, queue, tail).also { tail = it }
                if (x + 1 < w) add(p + 1, foreground, seen, queue, tail).also { tail = it }
                if (y > 0) add(p - w, foreground, seen, queue, tail).also { tail = it }
                if (y + 1 < h) add(p + w, foreground, seen, queue, tail).also { tail = it }
            }
            val bw = maxX - minX + 1
            val bh = maxY - minY + 1
            if (points.size >= 18 && bw >= 3 && bh >= 8 && bw <= w / 2 && bh <= h / 2) {
                output += Component(
                    points.map { p -> ((p % w) + region.left) to ((p / w) + region.top) },
                    Rect(minX + region.left, minY + region.top, maxX + region.left + 1, maxY + region.top + 1),
                    red > points.size / 2
                )
            }
        }
        return output
    }

    private fun add(index: Int, fg: BooleanArray, seen: BooleanArray, queue: IntArray, tail: Int): Int {
        if (!seen[index] && fg[index]) {
            seen[index] = true
            queue[tail] = index
            return tail + 1
        }
        return tail
    }

    private fun mergeForTen(parts: List<Component>): List<Component> {
        val merged = mutableListOf<Component>()
        for (a in parts) for (b in parts) {
            if (a === b || a.mostlyRed != b.mostlyRed) continue
            val ah = a.bounds.height(); val bh = b.bounds.height()
            if (min(ah, bh) < max(ah, bh) * 0.65f) continue
            val gap = b.bounds.left - a.bounds.right
            if (gap !in 0..max(10, ah / 2)) continue
            if (abs(a.bounds.centerY() - b.bounds.centerY()) > max(ah, bh) / 3) continue
            merged += Component(a.points + b.points, union(a.bounds, b.bounds), a.mostlyRed)
        }
        return merged
    }

    private fun <T> best(shape: Shape, templates: Map<T, Shape>): Pair<T, Float>? = templates
        .map { (label, template) -> label to similarity(shape, template) }
        .maxByOrNull { it.second }

    private fun similarity(a: Shape, b: Shape): Float {
        var best = 0f
        for (dy in -2..2) for (dx in -2..2) {
            var overlap = 0
            var total = 0
            for (y in 0 until SHAPE_H) for (x in 0 until SHAPE_W) {
                val av = a.bits[y * SHAPE_W + x]
                val bx = x + dx; val by = y + dy
                val bv = bx in 0 until SHAPE_W && by in 0 until SHAPE_H && b.bits[by * SHAPE_W + bx]
                if (av || bv) total++
                if (av && bv) overlap++
            }
            if (total > 0) best = max(best, overlap.toFloat() / total)
        }
        return best
    }

    private fun shapeFromResource(context: Context, id: Int, lower: Boolean, keepTwo: Boolean = false): Shape {
        val bitmap = BitmapFactory.decodeResource(context.resources, id)
        val fromY = if (lower) bitmap.height / 4 else 0
        val toY = if (lower) bitmap.height else (bitmap.height * 9 / 10)
        val points = mutableListOf<Pair<Int, Int>>()
        for (y in fromY until toY) for (x in 0 until bitmap.width) {
            if (isInk(bitmap.getPixel(x, y))) points += x to y
        }
        val comps = split(points)
        val selected = if (keepTwo) comps.sortedByDescending { it.size }.take(2).flatten()
            else comps.maxByOrNull { it.size }.orEmpty()
        return normalize(selected)
    }

    private fun split(points: List<Pair<Int, Int>>): List<List<Pair<Int, Int>>> {
        val set = points.toMutableSet()
        val groups = mutableListOf<List<Pair<Int, Int>>>()
        while (set.isNotEmpty()) {
            val queue = ArrayDeque<Pair<Int, Int>>()
            val group = mutableListOf<Pair<Int, Int>>()
            val seed = set.first()
            queue.addLast(seed)
            set.remove(seed)
            while (queue.isNotEmpty()) {
                val p = queue.removeFirst(); group += p
                val neighbors = listOf(p.first - 1 to p.second, p.first + 1 to p.second, p.first to p.second - 1, p.first to p.second + 1)
                for (n in neighbors) if (set.remove(n)) queue.addLast(n)
            }
            groups += group
        }
        return groups
    }

    private fun Component.shape() = normalize(points)

    private fun normalize(points: List<Pair<Int, Int>>): Shape {
        if (points.isEmpty()) return Shape(BooleanArray(SHAPE_W * SHAPE_H))
        val minX = points.minOf { it.first }; val maxX = points.maxOf { it.first }
        val minY = points.minOf { it.second }; val maxY = points.maxOf { it.second }
        val bw = max(1, maxX - minX + 1); val bh = max(1, maxY - minY + 1)
        val bits = BooleanArray(SHAPE_W * SHAPE_H)
        for ((x, y) in points) {
            val nx = ((x - minX) * (SHAPE_W - 1) / bw).coerceIn(0, SHAPE_W - 1)
            val ny = ((y - minY) * (SHAPE_H - 1) / bh).coerceIn(0, SHAPE_H - 1)
            bits[ny * SHAPE_W + nx] = true
        }
        return Shape(bits)
    }

    private fun distancePenalty(a: Rect, b: Rect): Float = abs(a.centerY() - b.centerY()) / 500f
    private fun sameSpot(a: Rect, b: Rect) = abs(a.centerX() - b.centerX()) < 28 && abs(a.centerY() - b.centerY()) < 45
    private fun union(a: Rect, b: Rect) = Rect(min(a.left, b.left), min(a.top, b.top), max(a.right, b.right), max(a.bottom, b.bottom))

    private data class Shape(val bits: BooleanArray)
    private data class Component(val points: List<Pair<Int, Int>>, val bounds: Rect, val mostlyRed: Boolean)
    private data class ClassifiedRank(val component: Component, val rank: Rank, val score: Float)
    private data class ClassifiedSuit(val component: Component, val suit: Suit, val score: Float)

    companion object {
        private const val SHAPE_W = 28
        private const val SHAPE_H = 44

        private fun isInk(color: Int): Boolean {
            val r = Color.red(color); val g = Color.green(color); val b = Color.blue(color)
            val dark = r < 105 && g < 105 && b < 105
            val red = r > 145 && r > g * 1.45f && r > b * 1.25f
            return dark || red
        }

        private fun isRed(color: Int): Boolean {
            val r = Color.red(color); val g = Color.green(color); val b = Color.blue(color)
            return r > 145 && r > g * 1.45f && r > b * 1.25f
        }
    }
}
