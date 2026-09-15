package com.example.durakassistant.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.example.durakassistant.game.Observation
import com.example.durakassistant.game.TablePair
import com.example.durakassistant.game.TurnPhase
import kotlin.math.abs

class FrameAnalyzer(context: Context) {
    private val glyphs = GlyphMatcher(context)
    private val deckCounter = DeckCounter(context)

    fun analyze(frame: Bitmap): Observation {
        val w = frame.width; val h = frame.height
        val hand = glyphs.findCards(frame, Realme16Profile.hand(w, h))
        val tableDetections = glyphs.findCards(frame, Realme16Profile.table(w, h))
        val trump = glyphs.findTrumpSuit(frame, Realme16Profile.trump(w, h))
        val deck = deckCounter.read(frame, Realme16Profile.deckCounter(w, h))
        val table = pairTable(tableDetections)
        val userCanAct = hasRedActionCue(frame, Realme16Profile.actionCue(w, h))
        val confidenceParts = hand.map { it.confidence } + tableDetections.map { it.confidence } + listOfNotNull(trump?.second, deck?.second)
        val confidence = if (confidenceParts.isEmpty()) 0f else confidenceParts.average().toFloat()
        val phase = when {
            !userCanAct -> TurnPhase.WAIT
            table.isEmpty() -> TurnPhase.ATTACK
            table.any { it.defense == null } -> TurnPhase.DEFEND
            else -> TurnPhase.ATTACK
        }
        return Observation(
            hand = hand.map { it.card }.toSet(),
            table = table,
            trump = trump?.first,
            deckCount = deck?.first,
            opponentCount = null,
            phase = phase,
            confidence = confidence
        )
    }

    private fun hasRedActionCue(bitmap: Bitmap, region: Rect): Boolean {
        var red = 0
        var sampled = 0
        for (y in region.top until region.bottom step 3) for (x in region.left until region.right step 3) {
            val c = bitmap.getPixel(x, y)
            val r = Color.red(c); val g = Color.green(c); val b = Color.blue(c)
            if (r > 175 && r > g * 1.55f && r > b * 1.35f) red++
            sampled++
        }
        return sampled > 0 && red.toFloat() / sampled > 0.006f
    }

    private fun pairTable(cards: List<CardDetection>): List<TablePair> {
        val remaining = cards.toMutableList()
        val pairs = mutableListOf<TablePair>()
        while (remaining.isNotEmpty()) {
            val attack = remaining.removeAt(0)
            val defense = remaining
                .filter { it.bounds.top >= attack.bounds.top - 20 }
                .minByOrNull { abs(it.bounds.centerX() - attack.bounds.centerX()) + abs(it.bounds.top - attack.bounds.top) }
                ?.takeIf { abs(it.bounds.centerX() - attack.bounds.centerX()) < 135 && abs(it.bounds.top - attack.bounds.top) < 150 }
            if (defense != null) remaining.remove(defense)
            pairs += TablePair(attack.card, defense?.card)
        }
        return pairs
    }
}
