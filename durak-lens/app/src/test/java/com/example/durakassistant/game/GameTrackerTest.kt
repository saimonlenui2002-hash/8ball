package com.example.durakassistant.game

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GameTrackerTest {
    private val nineSpades = Card(Rank.NINE, Suit.SPADES)
    private val tenSpades = Card(Rank.TEN, Suit.SPADES)
    private val aceHearts = Card(Rank.ACE, Suit.HEARTS)

    @Test
    fun coveredTableGoesToDiscard() {
        val tracker = GameTracker()
        tracker.accept(observation(table = listOf(TablePair(nineSpades, tenSpades))))

        val state = tracker.accept(observation())

        assertEquals(setOf(nineSpades, tenSpades), state.discarded)
        assertTrue(state.opponentTaken.isEmpty())
    }

    @Test
    fun uncoveredTableMovingIntoMyHandIsNotAssignedToOpponent() {
        val tracker = GameTracker()
        tracker.accept(observation(table = listOf(TablePair(nineSpades))))
        tracker.accept(observation(hand = setOf(nineSpades, aceHearts)))

        val state = tracker.accept(observation(hand = setOf(nineSpades, aceHearts)))

        assertTrue(state.opponentTaken.isEmpty())
        assertFalse(nineSpades in state.knownOpponent)
    }

    @Test
    fun uncoveredTableMissingFromMyHandIsAssignedToOpponent() {
        val tracker = GameTracker()
        tracker.accept(observation(table = listOf(TablePair(nineSpades))))
        repeat(4) { tracker.accept(observation(hand = setOf(aceHearts))) }

        val state = tracker.current()

        assertTrue(nineSpades in state.opponentTaken)
        assertTrue(nineSpades in state.knownOpponent)
    }

    @Test
    fun emptyDeckMakesRemainingOpponentCardsExact() {
        val tracker = GameTracker()
        val myHand = setOf(nineSpades, tenSpades, aceHearts)

        val state = tracker.accept(observation(hand = myHand, deck = 0))

        assertEquals(Deck24.cards - myHand, state.knownOpponent)
        assertEquals(state.knownOpponent.size, state.opponentCount)
    }

    private fun observation(
        hand: Set<Card> = setOf(aceHearts),
        table: List<TablePair> = emptyList(),
        deck: Int = 12
    ) = Observation(
        hand = hand,
        table = table,
        trump = Suit.CLUBS,
        deckCount = deck,
        opponentCount = null,
        phase = TurnPhase.UNKNOWN,
        confidence = 0.9f
    )
}
