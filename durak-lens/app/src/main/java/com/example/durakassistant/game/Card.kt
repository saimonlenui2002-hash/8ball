package com.example.durakassistant.game

enum class Suit(val symbol: String, val red: Boolean) {
    CLUBS("♣", false), DIAMONDS("♦", true), HEARTS("♥", true), SPADES("♠", false)
}

enum class Rank(val strength: Int, val label: String) {
    NINE(9, "9"), TEN(10, "10"), JACK(11, "В"), QUEEN(12, "Д"), KING(13, "К"), ACE(14, "Т")
}

data class Card(val rank: Rank, val suit: Suit) {
    override fun toString(): String = "${rank.label}${suit.symbol}"
}

object Deck24 {
    val cards: Set<Card> = Suit.entries.flatMap { suit -> Rank.entries.map { Card(it, suit) } }.toSet()
}
