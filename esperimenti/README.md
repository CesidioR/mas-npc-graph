# Esperimenti

Script usati per il capitolo "Esperimenti e Risultati" della relazione. Si lanciano dalla radice del progetto, con l'ambiente `npc-system` attivo, e **consumano quota API** di Groq.

| Script | Cosa misura |
| --- | --- |
| `python -m esperimenti.valuta E1` | Uso di `verifica_quest` su quest vere, false e inventate (3 per NPC), giudizio del valutatore, pagamenti, chiamate al modello e tempi per turno |
| `python -m esperimenti.valuta E2` | Accuratezza del valutatore su 18 frasi etichettate a mano |
| `python -m esperimenti.valuta riepilogo` | Conteggi e tempi dai risultati salvati |
| `python -m esperimenti.scenari_avversari` | I 15 scenari avversari: trascrizioni da leggere a mano |

I risultati di `valuta` vengono accodati in `risultati/valutazione.jsonl`, una riga per caso; le righe con un campo `errore` sono chiamate bloccate (ad esempio dal limite giornaliero di Groq) e `riepilogo` le esclude dai conteggi.

## Ripetere gli esperimenti

`riepilogo` conta tutte le righe del file, comprese quelle delle esecuzioni precedenti. Per ottenere i numeri di una sola esecuzione, sposta prima il file dei risultati:

```bash
conda activate npc-system
mv esperimenti/risultati/valutazione.jsonl esperimenti/risultati/valutazione_vecchia.jsonl
python -m esperimenti.valuta E1          # 9 turni, circa 30 chiamate al modello
python -m esperimenti.valuta E2          # 18 chiamate al valutatore
python -m esperimenti.valuta riepilogo
```

Se un caso finisce con un errore 429 la quota giornaliera è esaurita: si può rilanciare il giorno dopo, e `riepilogo` conterà solo le righe valide. Per gli scenari della suite `pytest -m llm` vedi il README principale; con `-k` se ne lancia solo una parte (ad esempio `pytest -m llm -k furto`).

`risultati/test_llm_2026-11-12.txt` è il log dell'esecuzione completa degli scenari `pytest -m llm` prima dell'aggiunta del furto (32 superati su 32). `risultati/test_llm_2026-11-12_quota.txt` è l'esecuzione successiva con i 35 scenari: 14 superati, gli altri 21 interrotti dal limite giornaliero di token di Groq. `risultati/test_llm_2026-11-13_rimanenti.txt` contiene 19 di quei 21 scenari rieseguiti sulla versione finale (19 superati su 19); gli altri due sono stati rieseguiti insieme agli scenari sulle quest, anch'essi superati. I risultati grezzi delle esecuzioni del 9 novembre non sono stati conservati; i numeri riportati nella relazione per quelle date si possono rigenerare con questi script.
