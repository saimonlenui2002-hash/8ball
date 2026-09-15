package com.example.durakassistant.vision

import com.example.durakassistant.game.Card
import com.example.durakassistant.game.Observation
import com.example.durakassistant.game.TablePair

/**
 * Rejects one-frame recognitions caused by card dealing and cover animations.
 * The hand uses a five-frame majority; the table uses a shorter three-frame
 * majority so move advice still reacts quickly.
 */
class ObservationStabilizer {
    private val history = ArrayDeque<Observation>()
    private var confirmedDeck: Int? = null
    private var newDealVotes = 0

    fun reset() {
        history.clear()
        confirmedDeck = null
        newDealVotes = 0
    }

    fun offer(observation: Observation): Observation? {
        history.addLast(observation)
        while (history.size > HAND_WINDOW) history.removeFirst()
        if (history.size < TABLE_WINDOW) return null

        val handFrames = history.toList()
        val tableFrames = handFrames.takeLast(TABLE_WINDOW)
        val handMode=handFrames.filter{it.handComplete}.groupBy{it.hand}.maxByOrNull{it.value.size}
        val hand=handMode?.key?:observation.hand
        val handComplete=handMode!=null && handMode.value.size>=3 && observation.handComplete && observation.hand==hand
        val tableCards = majorityCards(tableFrames.map { it.tableCards }, 2)
        val table = stableTable(tableFrames, tableCards)
        val trump = handFrames.mapNotNull { it.trump }
            .groupingBy { it }.eachCount().maxByOrNull { it.value }
            ?.takeIf { it.value >= requiredVotes(handFrames.size) }?.key

        val deckMode = handFrames.mapNotNull { it.deckCount }
            .groupingBy { it }.eachCount().maxByOrNull { it.value }
            ?.takeIf { it.value >= 3 }?.key
        if (deckMode == 12 && confirmedDeck != null && confirmedDeck!! <= 2 &&
            table.isEmpty() && hand.size in 4..8
        ) {
            newDealVotes++
            if (newDealVotes >= NEW_DEAL_VOTES) confirmedDeck = 12
        } else {
            newDealVotes = 0
            if (deckMode != null && (confirmedDeck == null || deckMode <= confirmedDeck!!)) {
                confirmedDeck = deckMode
            }
        }

        return observation.copy(
            hand = hand,
            table = table,
            trump = trump,
            // A previous confirmed value must not turn an unread live frame into
            // positive evidence of an empty deck or a new deal.
            deckCount = if(observation.deckCount==confirmedDeck)confirmedDeck else null,
            handComplete = handComplete,
            tableComplete = observation.tableComplete && tableFrames.count{it.tableCards==tableCards && it.tableComplete}>=2,
            confidence = handFrames.map { it.confidence }.average().toFloat()
        )
    }

    private fun stableTable(frames: List<Observation>, confirmed: Set<Card>): List<TablePair> {
        if (confirmed.isEmpty()) return emptyList()
        val reference = frames.asReversed().firstOrNull { frame ->
            confirmed.all { it in frame.tableCards }
        } ?: frames.last()
        return reference.table.mapNotNull { pair ->
            if (pair.attack !in confirmed) null
            else TablePair(pair.attack, pair.defense?.takeIf { it in confirmed })
        }
    }

    private fun majorityCards(frames: List<Set<Card>>, votes: Int): Set<Card> = frames
        .flatMap { it }
        .groupingBy { it }
        .eachCount()
        .filterValues { it >= votes }
        .keys

    private fun requiredVotes(size: Int): Int = size / 2 + 1

    companion object {
        private const val HAND_WINDOW = 5
        private const val TABLE_WINDOW = 3
        private const val NEW_DEAL_VOTES = 5
    }
}
