package com.example.durakassistant.game

class GameTracker {
    private var state = GameKnowledge()
    private var pendingClear: PendingClear? = null

    @Synchronized
    fun reset() {
        state = GameKnowledge()
        pendingClear = null
    }

    @Synchronized
    fun current(): GameKnowledge = state

    @Synchronized
    fun accept(observation: Observation): GameKnowledge {
        val previous = state
        val observedTable = observation.tableCards

        // A stable return to 12 cards after an exhausted deck is a new deal.
        if (observation.deckCount == 12 && previous.deckCount <= 2 &&
            observedTable.isEmpty() && observation.hand.size in 4..8
        ) {
            pendingClear = null
            state = freshState(observation)
            return state
        }

        val previousTable = previous.table.flatMap { listOfNotNull(it.attack, it.defense) }.toSet()
        val deckCount = observation.deckCount?.let { minOf(previous.deckCount, it) } ?: previous.deckCount
        var discarded = previous.discarded
        var knownOpponent = previous.knownOpponent
        var opponentTaken = previous.opponentTaken
        var lastEvent = previous.lastEvent

        // Resolve an uncovered cleared table only after the hand recognizer had
        // enough frames to see whether those cards moved into the player's hand.
        pendingClear?.let { pending ->
            pending.age++
            val returnedToPlayer = pending.cards.intersect(observation.hand)
            when {
                returnedToPlayer.isNotEmpty() -> {
                    lastEvent = "Вы взяли: ${format(pending.cards)}"
                    pendingClear = null
                }
                pending.age >= TAKE_CONFIRM_FRAMES -> {
                    knownOpponent += pending.cards
                    opponentTaken += pending.cards
                    lastEvent = "Соперник взял: ${format(pending.cards)}"
                    pendingClear = null
                }
            }
        }

        if (previousTable.isNotEmpty() && observedTable.isEmpty() && pendingClear == null) {
            val fullyCovered = previous.table.all { it.defense != null }
            if (fullyCovered) {
                discarded += previousTable
                lastEvent = "Ушли в биту: ${format(previousTable)}"
            } else {
                pendingClear = PendingClear(previousTable)
                lastEvent = "Проверяю, кто взял карты…"
            }
        }

        // A unique card cannot be in the hand, on the table and in another
        // tracked zone simultaneously. Prefer visible cards over inferred ones.
        val currentTable = observedTable - discarded
        val hand = observation.hand - currentTable - discarded
        knownOpponent = (knownOpponent - currentTable - hand).intersect(Deck24.cards - discarded)

        val impossible = hand + currentTable + discarded
        val candidates = Deck24.cards - impossible
        var opponentCount = observation.opponentCount ?: (
            Deck24.cards.size - deckCount - hand.size - discarded.size - currentTable.size
        )
        opponentCount = opponentCount.coerceIn(knownOpponent.size, candidates.size)

        // Once the deck is empty, every card not visible in the player's hand,
        // on the table or in discard must be in the opponent's hand.
        if (deckCount == 0 && pendingClear == null) {
            knownOpponent = candidates
            opponentCount = candidates.size
            lastEvent = "Колода закончилась — рука соперника вычислена"
        }

        state = GameKnowledge(
            hand = hand,
            table = observation.table.filter { it.attack in currentTable }.map {
                TablePair(it.attack, it.defense?.takeIf { card -> card in currentTable })
            },
            trump = observation.trump ?: previous.trump,
            deckCount = deckCount.coerceIn(0, 12),
            opponentCount = opponentCount,
            knownOpponent = knownOpponent.intersect(candidates),
            possibleOpponent = candidates,
            discarded = discarded,
            opponentTaken = opponentTaken,
            lastEvent = lastEvent,
            phase = resolvePhase(observation),
            confidence = observation.confidence
        )
        return state
    }

    private fun freshState(observation: Observation): GameKnowledge {
        val tableCards = observation.tableCards
        val hand = observation.hand - tableCards
        val candidates = Deck24.cards - hand - tableCards
        val opponentCount = (Deck24.cards.size - 12 - hand.size - tableCards.size).coerceAtLeast(0)
        return GameKnowledge(
            hand = hand,
            table = observation.table,
            trump = observation.trump,
            deckCount = 12,
            opponentCount = opponentCount,
            possibleOpponent = candidates,
            lastEvent = "Началась новая партия",
            phase = resolvePhase(observation),
            confidence = observation.confidence
        )
    }

    private fun resolvePhase(observation: Observation): TurnPhase {
        if (observation.phase != TurnPhase.UNKNOWN) return observation.phase
        val attacks = observation.table.size
        val defenses = observation.table.count { it.defense != null }
        return when {
            attacks == 0 -> TurnPhase.ATTACK
            attacks > defenses -> TurnPhase.DEFEND
            attacks == defenses -> TurnPhase.ATTACK
            else -> TurnPhase.WAIT
        }
    }

    private fun format(cards: Set<Card>): String = cards
        .sortedWith(compareBy({ it.suit.ordinal }, { it.rank.strength }))
        .joinToString(" ")

    private data class PendingClear(val cards: Set<Card>, var age: Int = 0)

    companion object {
        private const val TAKE_CONFIRM_FRAMES = 3
    }
}
