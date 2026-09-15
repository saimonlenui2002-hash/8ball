package com.example.durakassistant.game

data class Advice(val title: String, val detail: String, val card: Card? = null)

class MoveAdvisor {
    fun advise(state: GameKnowledge): Advice {
        if (state.hand.isEmpty()) return Advice("Ожидание", "Рука ещё не распознана")
        val trump = state.trump ?: return Advice("Ожидание", "Определяю козырь")
        return when (state.phase) {
            TurnPhase.DEFEND -> defend(state, trump)
            TurnPhase.ATTACK -> attack(state, trump)
            TurnPhase.WAIT -> Advice("Ход соперника", "Отслеживаю сыгранные карты")
            TurnPhase.UNKNOWN -> Advice("Анализ", "Определяю состояние хода")
        }
    }

    private fun defend(state: GameKnowledge, trump: Suit): Advice {
        val open = state.table.lastOrNull { it.defense == null }?.attack
            ?: return Advice("Подкинуть", "Все текущие карты покрыты")

        // In transfer durak a matching rank may transfer an untouched attack.
        val canTransfer = state.table.none { it.defense != null }
        val transfer = state.hand
            .filter { canTransfer && it.rank == open.rank }
            .minByOrNull { cost(it, trump) }
        if (transfer != null && state.opponentCount > state.table.size) {
            return Advice("Перевести ${transfer}", "Сохраняет старшие карты", transfer)
        }

        val cover = state.hand
            .filter { beats(it, open, trump) }
            .minByOrNull { cost(it, trump) }
        return if (cover != null) {
            Advice("Покрыть ${cover}", "Самая дешёвая подходящая карта", cover)
        } else {
            Advice("Брать", "Подходящей карты не найдено")
        }
    }

    private fun attack(state: GameKnowledge, trump: Suit): Advice {
        val tableRanks = state.table.flatMap { listOfNotNull(it.attack.rank, it.defense?.rank) }.toSet()
        val legal = if (tableRanks.isEmpty()) state.hand else state.hand.filter { it.rank in tableRanks }.toSet()
        if (legal.isEmpty()) return Advice("Бито", "Подходящего достоинства для подкидывания нет")

        val choice = legal.minByOrNull { card ->
            var score = cost(card, trump)
            if (state.knownOpponent.any { it.suit == card.suit && it.rank.strength > card.rank.strength }) score += 5
            score
        }!!
        return Advice("Ходить ${choice}", "Минимальная потеря силы руки", choice)
    }

    private fun beats(defense: Card, attack: Card, trump: Suit): Boolean = when {
        defense.suit == attack.suit -> defense.rank.strength > attack.rank.strength
        defense.suit == trump && attack.suit != trump -> true
        else -> false
    }

    private fun cost(card: Card, trump: Suit): Int = card.rank.strength + if (card.suit == trump) 20 else 0
}
