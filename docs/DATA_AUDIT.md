# Datan laatuauditointi — Prompt Optimizer

**Päivämäärä:** 2025-05-14  
**Projekti:** QLoRA-finetuning Qwen2.5-3B-Instruct-mallille (prompt-optimizer)

---

## A) Perustilastot

| Tiedosto | Rivejä (paria) |
|---|---:|
| `train.jsonl` | 946 |
| `val.jsonl` | 118 |
| `test.jsonl` | 119 |
| `cleaned_data.jsonl` | 1 183 |
| `raw/generated_data.jsonl` | 683 |
| `raw/augmented_data.jsonl` | 3 344 |
| `raw/seed_data.jsonl` | 25 |

**Jakosuhde (train/val/test):** 80%/10%/10% — oikeaoppinen jako.

**Huomio:** `cleaned_data.jsonl` (1 183 riviä) sisältää kaikki train+val+test -datat yhteensä (946+118+119 = 1 183). Näin kuuluukin olla.

---

## B) Kategoriajakauma

### Train (946 paria)

| Kategoria | Määrä | Osuus |
|---|---:|---:|
| coding | 165 | 17,4% |
| writing | 160 | 16,9% |
| analysis | 148 | 15,6% |
| brainstorming | 108 | 11,4% |
| editing | 83 | 8,8% |
| q_and_a | 82 | 8,7% |
| instruction | 79 | 8,4% |
| roleplay | 56 | 5,9% |
| translation | 37 | 3,9% |
| summarization | 22 | 2,3% |
| mixed | 6 | 0,6% |

### Val (118) ja Test (119)

Jakaukset seuraavat suurin piirtein trainin suhteita, mutta **mixed**-kategoria puuttuu validaatiodatasta kokonaan (0 kappaletta) ja **summarization** (2 kpl) ja **translation** (2–3 kpl) ovat hyvinohontaisia molemmissa.

**Arvio:** Kategoriat ovat **epätasapainoisia**. Viisi ylintä kategoriaa (coding, writing, analysis, brainstorming, editing) kattavat 79,8% datasta. Vähimmäiset kategoriat (summarization, mixed, translation) ovat alle 7% yhdessä. Tämä voi aiheuttaa mallin huonompaa suoriutumista harvinaisissa kategorioissa.

---

## C) Formaattivalidointi

### Train/Val/Test — Muoto

Kaikki päädatat käyttävät rakennetta:

```json
{
  "instruction": "Optimize the following prompt...",
  "input": "<epäselvä prompti>",
  "output": "<optimoitu prompti>",
  "category": "<kategoria>",
  "system": "<system prompt>"
}
```

- **Kaikki 946+118+119 = 1 183 riviä** ovat validia JSON:ia — ei malformed-rivejä
- **Ei puuttuvia kenttiä** (instruction, input, output, category, system kaikki täytetty)
- **Instruction-kenttä** on identtinen kaikissa riveissä: *"Optimize the following prompt to be clear, specific, and effective. Preserve the original intent while adding structure, context, and constraints where appropriate."*
- **System-kenttä** on identtinen kaikissa riveissä: *"You are a prompt engineering expert that transforms vague, underspecified prompts into clear, well-structured, and effective prompts..."*

### Cleaned/Raw — Eri kenttänimet

`cleaned_data.jsonl`, `raw/generated_data.jsonl`, `raw/augmented_data.jsonl` ja `raw/seed_data.jsonl` käyttävät eri kenttänimet:

```json
{
  "vague": "<epäselvä prompti>",
  "optimized": "<optimoitu prompti>",
  "category": "<kategoria>"
}
```

Näistä puuttuvat `instruction`- ja `system`-kentät, ja `input`→`vague`, `output`→`optimized`. Tämä täytyy huomioida preprocessoinnissa.

---

## D) Duplikaattianalyysi

### Tarkat duplikaatit

- **Train:** 0 tarkkaa input-duplikaattia (946 uniikkia / 946)
- **Train↔Val:** 0 overlapia
- **Train↔Test:** 0 overlapia
- **Val↔Test:** 0 overlapia

Datojen välinen vuototarkistus (data leakage) **läpäistään** — ei yhteisiä promptteja eri splitien välillä.

### Lähiduplikaatit

Löydettiin **553 lähes identtistä paria** (SequenceMatcher-ratio > 0.85) train-datan sisällä. Tyypilliset syyt:

1. **Pisteen poisto/lisäys**: `"tell me about OAuth 2.0"` vs `"tell me about OAuth 2.0."` (ratio 0.98)
2. **Pieniin sanoihin muutos**: `"give me ideas for a mobile app."` vs `"creative ideas for a mobile app."` (ratio 0.86)
3. **Pyyntösanan muutos**: `"translate this to French."` vs `"translate this to Finnish."` (ratio 0.86)

**Arvio:** Noin 553 lähes identtistä paria 946 rivin datassa on merkittävä määrät. Arviolta 100–200 uniikkia inputia voi olla kytköksissä toisiinsa. Tämä voi aiheuttaa:
- Mallin ylioppimista (overfitting) samankaltaisiin inputteihin
- Harhaanjohtavaa evaluaatiota, jos läheiduplikaatit sattuvat eri spliteihin (ei onneksi näin tässä)

**Suositus:** Poista tai deduplikoi near-duplikaatit ennen lopullista opetusta.

---

## E) Laatutarkastukset

### Pituustilastot

| Mittari | Input | Output |
|---|---:|---:|
| **Train — min** | 4 merkkiä | 38 merkkiä |
| **Train — max** | 60 merkkiä | 359 merkkiä |
| **Train — avg** | 31,7 | 233,3 |
| **Val — min** | 16 | 61 |
| **Val — max** | 63 | 352 |
| **Val — avg** | 31,9 | 240,0 |
| **Test — min** | 13 | 61 |
| **Test — max** | 58 | 359 |
| **Test — avg** | 31,1 | 231,3 |

- **Ei tyhjiä merkkijonoja** (null/empty): Kaikissa kentissä arvoja
- **Ei identtisiä input→output -pareja**: Kaikissa tapauksissa output on pidempi ja optimoitu
- **Ei regression tapauksia**: Kaikissa 946 train-parissa output on pidempi kuin input (0 tapausta jossa output < input)

### Merkistöongelmat

Löydettiin **24 tapausta em-dash-merkistä (U+2014 —)** train-datan output-kentissä. Nämä ovat täysin normaaleja typografisia merkkejä eivätkä ole ongelmallisia, mutta on hyvä varmistaa että tokenizer tukee niitä.

Ei muita ei-ASCII-ongelmia havaittu.

### Minimi input-length (4 merkkiä)

Lyhin input on vain 4 merkkiä. Tämä voi tuottaa epävarmoja optimointeja. Suositus: tarkista että lyhyet inputit tuottavat järkeviä outputteja.

---

## F) Kategorakohtainen analyysi

| Kategoria | n (train) | Input avg | Output avg | Output/Input-suhde |
|---|---:|---:|---:|---:|
| writing | 160 | 40,9 | 100,1 | 2,4x |
| coding | 165 | 29,1 | 203,2 | 7,0x |
| analysis | 148 | 38,0 | 258,4 | 6,8x |
| brainstorming | 108 | 34,3 | 288,4 | 8,4x |
| editing | 83 | 23,7 | 285,7 | 12,0x |
| q_and_a | 82 | 22,4 | 250,1 | 11,2x |
| instruction | 79 | 31,6 | 324,6 | 10,3x |
| roleplay | 56 | 28,3 | 293,7 | 10,4x |
| translation | 37 | 25,8 | 219,1 | 8,5x |
| summarization | 22 | 18,1 | 252,5 | 14,0x |
| mixed | 6 | 19,2 | 298,5 | 15,6x |

**Huomioita:**

1. **Writing** erottuu joukosta: sen output/input-suhde on vain 2,4x kun muut kategoriat ovat 6–16x. Writing-kategorian outputit ovat selvästi lyhyempiä (avg 100 merkkiä) vs. muut kategoriat (200–325 merkkiä). Tämä saattaa heijastaa dataongelmaa — writing-optimoinnit voivat olla liian niukkoja.

2. **Summarization ja mixed** ovat hyvin harvassa (22 ja 6 kpl trainissa). Malli tuskin oppii näitä kategorioita luotettavasti.

3. **Input-pituudet** vaihtelevat merkittävästi kategorioittain: summarization (18,1) ja mixed (19,2) ovat lyhyimpiä, writing (40,9) pisimpiä. Tämä on luonnollista, mutta vaihtelu voi vaikeuttaa mallin oppimista.

---

## G) Suositukset

### KRIITTINEN — Korjaa ennen opetusta

1. **Lähiduplikaattien poisto**: 553 lähes identtistä paria (~58% datasta on jossain lähes identtisessä suhteessa) on liian paljon. Suositus: käytä deduplikointia esim. MinHash tai treshold-basierattu SequenceMatcher-ratio > 0.90 ja poista toinen kopio. Tavoite: < 5% near-duplicateja.

2. **Kategoriatasapaino**: Poista `mixed`-kategoria (vain 6 kpl) tai yhdistä se lähimpään kategoriaan. Tarkista `summarization` (22 kpl) ja `translation` (37 kpl) — harkitse ylisampplausta (oversampling) tai näiden kategorioiden yhdistämistä data-augmentaatiolla.

### TÄRKEÄ — Parantaakseen mallin laatua

3. **Writing-kategorian outputit ovat liian lyhyitä**: Keskimmäärin vain 100 merkkiä (vs. muiden kategorioiden 200–325). Tarkista onko writing-datan outputit jääneet vajaiksi — moni writing-optimointi näyttää lisäävän vain sanamäärän ja kohdeyleisön, ei muuta rakennetta.

4. **Poista mahdolliset inputit alle 10 merkillä**: Lyhyin input on 4 merkkiä — tarkista että nämä tuottavat mielekkäitä optimointeja.

5. **Field-nimien konsistenssi**: Raakadatoissa käytetään `vague`/`optimized`-kenttä nimiä, kun taas train/val/test käyttävät `input`/`output`. Varmista, että preprocessointi tekee tämän konversion oikein.

### HYVÄÄ — Mikä on kunnossa

- **Ei data vuotoa**: Train/Val/Test -spliteissä ei oleoverlapia — erinomainen
- **Ei tyhjiä arvoja**: Kaikilla riveillä on kaikki kentät — puhdas data
- **Ei identtisiä input→output -pareja**: Kaikissa tapauksissa malli oppii muunnoksen
- **Yksinkertainen ja konsistentti instruction/system-template**: Helpottaa finetuningia
- **Oikeaoppinen jakosuhde**: 80/10/10

---

## Yhteenveto

Datasetin perusrakenne on terve: oikeaoppinen jako, ei duplikaatteja splitien välillä, ei puuttuvia arvoja. Pääongelmat ovat:

1. **Liikaa lähiduplikaatteja** train-datan sisällä (~553 paria)
2. **Kategoriatasapaino epävakaa** (mixed: 6 kpl, summarization: 22 kpl)
3. **Writing-kategorian outputit poikkeavan lyhyitä** suhteessa muihin kategorioihin

Nämä korjattuna datasetti on finetuning-valmis.