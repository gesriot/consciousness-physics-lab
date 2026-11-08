# Preconfirmation implementation correction

The first implementation was committed as 92950d2 before generation. Development
was interrupted after three complete series and the first package of series 3.
Review found that the equal-volume link interventions included pre-notice records
whose target support was at or before the last successful sample. Such records
were already ineligible for G, so the edits did not change its candidate ranking,
but the intervention population did not match the registered eligible-record rule.

Before the stop request arrived, series 0's summary had been inspected: its band-0
fault scenarios had zero affected and zero false accepted claims, and G performed
zero recalls while T and A performed some. Therefore this is not described as a
correction made without seeing any outcomes. No distributions, seeds, selected
bands, exposure gates, onset, notice, budgets or hypotheses were changed.

The correction restricts both interventions to the identical records with target
support strictly after last success. Tests cover the equality boundary, identical
edited record IDs, future-record invariance and distinct sample/action chronology.
The interrupted artifacts and log are retained. The same registered development
seeds are restarted in development-complete after this separate correction commit.
Only development-complete supplies the committed freeze and confirmation choice.
