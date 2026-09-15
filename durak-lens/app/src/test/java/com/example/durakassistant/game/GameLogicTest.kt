package com.example.durakassistant.game

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GameLogicTest {
    private val advisor = MoveAdvisor()

    @Test
    fun `deck contains exactly 24 unique cards`() {
        assertEquals(24, Deck24.cards.size)
    }

    @Test
    fun `advisor uses lowest same-suit cover before trump`() {
        val state = GameKnowledge(
            hand = setOf(Card(Rank.TEN, Suit.CLUBS), Card(Rank.JACK, Suit.HEARTS)),
            table = listOf(TablePair(Card(Rank.NINE, Suit.CLUBS))),
            trump = Suit.HEARTS,
            phase = TurnPhase.DEFEND
        )
        assertEquals(Card(Rank.TEN, Suit.CLUBS), advisor.advise(state).card)
    }

    @Test
    fun `translation is preferred before covering`() {
        val state = GameKnowledge(
            hand = setOf(Card(Rank.NINE, Suit.DIAMONDS), Card(Rank.ACE, Suit.CLUBS)),
            table = listOf(TablePair(Card(Rank.NINE, Suit.CLUBS))),
            trump = Suit.SPADES,
            opponentCount = 5,
            phase = TurnPhase.DEFEND
        )
        assertTrue(advisor.advise(state).title.startsWith("Перевести"))
    }

    @Test
    fun `elimination reveals complete hand after deck is empty`() {
        val own = Deck24.cards.take(6).toSet()
        val table = Deck24.cards.drop(6).take(2).map { TablePair(it) }
        val discarded = Deck24.cards.drop(8).take(10).toSet()
        val remaining = Deck24.cards - own - table.map { it.attack }.toSet() - discarded
        assertEquals(6, remaining.size)
    }
}
