# Logit-lens readout of grpo3 steering vectors

Organism: gpt-oss-120b + grpo3 LoRA (GRPO-trained CoT controllability). Vectors at decoder-block outputs of layers 9, 18, 27 (of 36). Sources: `logit_lens.json`, `vector_stats.json` (this directory), produced by `steering/vectors.py` and `steering/readout.py`. All numbers below were computed from those two files.

- **A** = mean_org - mean_base over 255,000 wikitext-103 tokens.
- **B** = mean_org - mean_base over 301,206 analysis-channel tokens of 600 grpo3 rollouts (same tokens teacher-forced through both models).
- **C_org / C_base** = mean(CoT tokens) - mean(user + final tokens), in the organism / in the base model. The non-CoT mean is token-weighted over 155,058 user and 1,800 final tokens, so it is 98.9% user-prompt tokens.
- **rand** = Gaussian direction (seed 0) scaled to |B| at that layer. **mu_base_wiki / mu_base_cot** = base model mean activations (controls).
- Lens = base final RMSNorm applied to the vector itself, then base `lm_head`, for +v and -v. Because the vector is normalised on its own, the readout depends only on direction, not on norm. Ranks are 0-indexed over 201,088 tokens.

## 1. Norms and cosines

| layer | hnorm | A/h | B/h | C_org/h | C_base/h | rand/h | cos A,B | cos B,C_org | cos B,C_base | cos C_org,C_base | cos B,rand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 1225 | 0.025 | 0.027 | 0.278 | 0.269 | 0.027 | 0.610 | 0.427 | 0.419 | 0.998 | -0.002 |
| 18 | 10707 | 0.043 | 0.092 | 0.392 | 0.355 | 0.092 | 0.321 | 0.620 | 0.501 | 0.985 | 0.003 |
| 27 | 44516 | 0.034 | 0.094 | 0.292 | 0.256 | 0.094 | 0.440 | 0.537 | 0.306 | 0.964 | 0.034 |

## 2. Top-12 tokens (+v / -v)

Tokens are Python-repr escaped (`\n`, `\u200b`, ...); a leading space is part of the token.

### Layer 9

| vector | sign | top-12 tokens |
|---|---|---|
| A | + | ` \u200b\u200b` ` Armour` ` War` `APS` `oche` ` Newly` ` Holy` ` Rock` ` acquaint` ` aps` ` Hand` `Far` |
| A | - | `<\|endoftext\|>` `<\|reserved_200016\|>` `/tag` `/extensions` `-cu` `/link` `cept` `/net` `itte` `cu` `/ref` `quis` |
| B | + | ` Weaver` ` principal` ` Werner` ` Down` ` Newly` ` Builders` ` Sullivan` ` Fen` `rush` ` Dai` ` +\r\n` ` DOWN` |
| B | - | `<\|endoftext\|>` `<\|reserved_200016\|>` ` LETTER` `LETTER` ` Hä` ` unconventional` `haa` `letter` `Prescription` `(letter` `Bro` `aja` |
| C_org | + | ` Pete` ` Werner` `ISE` ` Sawyer` `ché` `ovski` ` Syd` ` Zug` ` Swan` `\n\t\t\t\n` ` Jess` ` Sag` |
| C_org | - | ` unst` `.Br` ` Bro` ` honestly` ` Cran` ` unat` ` bro` `.br` ` blasted` ` ambitious` ` CRT` `Branch` |
| C_base | + | ` Pete` ` Werner` `ISE` ` Sawyer` `ovski` `ché` ` Swan` ` Syd` ` Sag` `\n\t\t\t\n` `Pete` ` Zug` |
| C_base | - | ` unst` `.Br` ` Bro` ` honestly` ` Cran` ` unat` ` blasted` ` bro` `.br` ` CRT` `Branch` ` ambitious` |
| rand | + | ` bof` ` Neder` ` Prices` ` inaccur` ` Ses` `rof` ` neglig` `abes` `yf` ` Suzuki` `Suz` ` numer` |
| rand | - | ` mak` ` édit` ` envelope` ` kitchens` `Mak` ` Mak` `oa` `#line` ` Arch` ` bound` ` lif` ` libraries` |
| mu_base_wiki | + | ` harness` ` hand` ` rational` ` technically` ` pl` ` rec` ` carried` `_` ` witnessed` ` headed` ` intern` `won` |
| mu_base_wiki | - | `flamm` ` grin` `quip` `nich` ` Tram` ` mow` `olat` ` Stops` ` Smarty` ` washer` `ctime` ` Tob` |
| mu_base_cot | + | ` double` ` bott` ` dropdown` ` technically` ` push` ` correlated` ` hand` ` ratio` ` rat` ` Double` ` equilibrium` ` circular` |
| mu_base_cot | - | ` superf` ` Piet` `obel` `inders` `flamm` `fod` `Dial` `quip` `-tip` `gow` `Fy` `yrus` |

### Layer 18

| vector | sign | top-12 tokens |
|---|---|---|
| A | + | `onden` `idgets` `ewo` `wach` `önt` `woon` ` reed` ` fuck` ` Sack` ` bangs` `-bank` `imbabwe` |
| A | - | ` to` ` about` `UC` `Thi` ` Thi` ` UC` ` US` ` thoughtfully` ` very` ` carefully` ` Chef` ` UT` |
| B | + | ` inherently` ` nay` ` nineteen` ` wasn` ` constantly` ` IRead` ` quite` ` ope` ` awhile` ` weren` ` ninety` ` essent` |
| B | - | `<\|endoftext\|>` `<\|reserved_200016\|>` ` "` `\n\n` `**\n\n` ` &` `:\n\n` ` Amb` ` Shar` ` Diff` ` PA` ` IS` |
| C_org | + | ` simplistic` ` yep` ` ()=>` ` infatti` ` simu` ` mase` ` upto` ` IRead` ` voila` ` n't` ` flatten` ` nope` |
| C_org | - | `\n` `‐` `**\n` `\xa0` `\u202f` `,\n` `\n\n` ` &` ` \n` `?\n` ` {\n` `\r\n` |
| C_base | + | ` yep` ` ()=>` ` voila` ` simplistic` ` infatti` ` megs` ` simu` ` mase` ` upto` ` thankfully` ` n't` ` negligible` |
| C_base | - | `\n` `‐` `**\n` `,\n` `\xa0` `\u202f` ` \n` ` {\n` `?\n` `_\n` `.\n` ` &` |
| rand | + | ` downstairs` ` Fight` ` sagitt` `ymi` ` substring` `_trait` `öld` ` theta` ` Beta` `.instances` `zimmer` ` GP` |
| rand | - | `make` ` Rocks` `j` ` make` ` jelly` ` reefs` ` rocks` `Tow` `successful` ` Retr` `/render` ` wrappers` |
| mu_base_wiki | + | `\n\n` `\n` ` ` `,` ` (` ` \n` `:` ` and` `-` `<\|endoftext\|>` `.` `/` |
| mu_base_wiki | - | `xtype` ` awo` `的网址` ` Centros` ` chup` ` IBase` `的邀请码` ` kiko` ` zogenaamde` ` pdata` ` piem` ` qualc` |
| mu_base_cot | + | ` (` ` ` `:` `?` ` =` `\n\n` `,` `.` `-` `;` `\n` ` and` |
| mu_base_cot | - | `"display` `"profile` `"group` `"AT` `cante` `directories` `"url` `timeouts` `administrator` `libraries` `pdev` `\tcan` |

### Layer 27

| vector | sign | top-12 tokens |
|---|---|---|
| A | + | ` ,\n\n` ` ,` ` :",` ` ,\n` ` .,` ` ."` ` ,\r\n` ` ,.` ` ;` ` 、` ` ),` ` )` |
| A | - | ` they'll` ` they're` ` they've` `—they` ` they'd` ` They've` ` They'll` `They're` ` They're` `:\n` ` they` `/conf` |
| B | + | ` eighty` ` ninety` ` forty` ` seventeen` ` fifty` ` youll` ` eighteen` ` saturday` ` fourteen` ` october` ` nineteen` ` youve` |
| B | - | ` '` ` "` ` ('` ` ("` `=>` `/ac` ` They're` `/sw` ` They've` `'=>'` `'=>` ` (§` |
| C_org | + | ` also` ` approx` ` maybe` ` plus` ` roughly` ` again` ` yep` ` indeed` `.` ` =` ` actually` ` a` |
| C_org | - | `-State` `-Core` ` Children's` `-Origin` ` Anthropology` ` Elabor` ` Mailing` `-Type` `TRANSFER` `breadcrumbs` `-De` `PROGRAM` |
| C_base | + | ` approx` ` also` ` maybe` ` =` ` same` ` ` ` again` `.` ` plus` ` a` ` sqrt` ` being` |
| C_base | - | `LETTER` `PROGRAM` ` Anthropology` `ABOUT` `TRANSFER` `GENERAL` `AUTHOR` ` Humanities` `breadcrumbs` `COMMENTS` `Declarations` `MASTER` |
| rand | + | `hra` ` Invitations` `cco` ` Serv` ` pra` ` invitation` `chairs` ` Harriet` `formes` `itsch` `noinspection` `hma` |
| rand | - | ` VK` ` detected` ` dwar` `SUN` ` pars` `PU` ` PE` ` mirror` `_go` ` sym` ` Bog` `_OVERRIDE` |
| mu_base_wiki | + | ` ` `\n\n` ` (` `,` `\xa0` ` ...` `...` ` \n` ` [` ` and` `-` ` the` |
| mu_base_wiki | - | `ENTO` ` draai` ` descuent` ` kandida` ` totaal` `ITOS` `的邀请码` ` zijde` ` kandidaat` ` meub` `osyo` `дамент` |
| mu_base_cot | + | ` ` ` (` `.` `:` `,` `?` ` =` ` a` ` -` `;` `\n\n` ` the` |
| mu_base_cot | - | `timeouts` `/Templates` `Advertisements` `"AT` `_unregister` `kamera` `-Identifier` `\tproperties` `\tinterface` `UILayout` `evas` `statuses` |

## 3. Case shares and probe ranks

`case_top200`: of the top-200 tokens, those that are alphabetic words of length >1 after stripping (n), and the share all-lowercase / all-uppercase. Subword fragments (`onden`, `idgets`) count as lowercase words, so a high lowercase share alone does not mean "lowercased words".

| layer | vector | + n | + lower | + upper | - n | - lower | - upper |
|---|---|---|---|---|---|---|---|
| 9 | A | 190 | 0.400 | 0.037 | 147 | 0.707 | 0.027 |
| 9 | B | 190 | 0.432 | 0.037 | 171 | 0.608 | 0.129 |
| 9 | C_org | 176 | 0.494 | 0.068 | 186 | 0.699 | 0.070 |
| 9 | C_base | 178 | 0.494 | 0.067 | 183 | 0.710 | 0.082 |
| 9 | rand | 183 | 0.656 | 0.033 | 168 | 0.673 | 0.036 |
| 9 | mu_base_wiki | 161 | 0.888 | 0.012 | 194 | 0.546 | 0.041 |
| 9 | mu_base_cot | 170 | 0.853 | 0.029 | 194 | 0.608 | 0.067 |
| 18 | A | 183 | 0.934 | 0.005 | 170 | 0.529 | 0.212 |
| 18 | B | 190 | 0.932 | 0.011 | 130 | 0.238 | 0.292 |
| 18 | C_org | 191 | 0.953 | 0.010 | 106 | 0.434 | 0.085 |
| 18 | C_base | 190 | 0.958 | 0.005 | 97 | 0.464 | 0.124 |
| 18 | rand | 183 | 0.672 | 0.077 | 177 | 0.621 | 0.034 |
| 18 | mu_base_wiki | 47 | 0.872 | 0.000 | 176 | 0.716 | 0.136 |
| 18 | mu_base_cot | 64 | 0.906 | 0.000 | 167 | 0.521 | 0.180 |
| 27 | A | 15 | 0.867 | 0.000 | 21 | 0.429 | 0.333 |
| 27 | B | 191 | 0.990 | 0.000 | 87 | 0.011 | 0.609 |
| 27 | C_org | 167 | 1.000 | 0.000 | 150 | 0.107 | 0.313 |
| 27 | C_base | 160 | 1.000 | 0.000 | 159 | 0.233 | 0.308 |
| 27 | rand | 191 | 0.565 | 0.016 | 175 | 0.537 | 0.246 |
| 27 | mu_base_wiki | 76 | 0.908 | 0.000 | 193 | 0.824 | 0.083 |
| 27 | mu_base_cot | 76 | 0.974 | 0.000 | 164 | 0.579 | 0.116 |

Probe ranks (0 = top of 201,088). The filler-word probes ` meow`, ` Meow`, ` MEOW`, ` woof`, ` quack` are multi-token in this tokenizer and were dropped by `readout.py`; ` beep` is the only filler-word probe present.

| layer | vec | sign | ` beep` | ` safe` | ` Safe` | ` the` | ` The` | ` THE` | ` we` | ` We` | ` WE` | ` so` | ` So` | ` SO` | `.` | ` answer` | ` Answer` | ` lowercase` | ` uppercase` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | A | + | 29804 | 28668 | 15997 | 15819 | 35647 | 183419 | 5649 | 4890 | 144987 | 123566 | 37609 | 6308 | 162967 | 139009 | 127777 | 199037 | 189246 |
| 9 | A | - | 171283 | 172419 | 185090 | 185268 | 165440 | 17668 | 195438 | 196197 | 56100 | 77521 | 163478 | 194779 | 38120 | 62078 | 73310 | 2050 | 11841 |
| 9 | B | + | 88785 | 49986 | 18713 | 77835 | 28084 | 173419 | 16282 | 26332 | 103824 | 99644 | 17895 | 8268 | 115894 | 138635 | 96111 | 200848 | 199261 |
| 9 | B | - | 112302 | 151101 | 182374 | 123252 | 173003 | 27668 | 184805 | 174755 | 97263 | 101443 | 183192 | 192819 | 85193 | 62452 | 104976 | 239 | 1826 |
| 9 | C_org | + | 48731 | 65108 | 27427 | 130006 | 36954 | 104317 | 69083 | 32353 | 12228 | 98292 | 57747 | 99902 | 173061 | 179361 | 178942 | 194496 | 198172 |
| 9 | C_org | - | 152356 | 135979 | 173660 | 71081 | 164133 | 96770 | 132004 | 168734 | 188859 | 102795 | 143340 | 101185 | 28026 | 21726 | 22145 | 6591 | 2915 |
| 9 | C_base | + | 49190 | 64146 | 28596 | 122945 | 38830 | 114265 | 66098 | 30987 | 13948 | 97163 | 59855 | 92426 | 172616 | 179563 | 179800 | 194286 | 198682 |
| 9 | C_base | - | 151897 | 136941 | 172491 | 78142 | 162257 | 86822 | 134989 | 170100 | 187139 | 103924 | 141232 | 108661 | 28471 | 21524 | 21287 | 6801 | 2405 |
| 9 | rand | + | 91260 | 67118 | 20677 | 99328 | 163880 | 128644 | 109499 | 50089 | 172626 | 185863 | 169760 | 143176 | 98656 | 146167 | 184346 | 182306 | 170658 |
| 9 | rand | - | 109827 | 133969 | 180410 | 101759 | 37207 | 72443 | 91588 | 150998 | 28461 | 15224 | 31327 | 57911 | 102431 | 54920 | 16741 | 18781 | 30429 |
| 9 | mu_base_wiki | + | 174676 | 11005 | 174170 | 1392 | 6644 | 85318 | 2699 | 7853 | 128979 | 8150 | 7251 | 3317 | 508 | 36201 | 115718 | 158107 | 28961 |
| 9 | mu_base_wiki | - | 26411 | 190082 | 26917 | 199695 | 194443 | 115769 | 198388 | 193234 | 72108 | 192937 | 193836 | 197770 | 200579 | 164886 | 85369 | 42980 | 172126 |
| 9 | mu_base_cot | + | 157668 | 1776 | 25140 | 1641 | 3855 | 82834 | 1430 | 294 | 54058 | 6488 | 1674 | 694 | 2467 | 27142 | 41850 | 156084 | 9046 |
| 9 | mu_base_cot | - | 43419 | 199311 | 175947 | 199446 | 197232 | 118253 | 199657 | 200793 | 147029 | 194599 | 199413 | 200393 | 198620 | 173945 | 159237 | 45003 | 192041 |
| 18 | A | + | 170964 | 169888 | 185848 | 199807 | 196952 | 165845 | 171248 | 189186 | 148570 | 200272 | 197855 | 195324 | 200949 | 199569 | 200590 | 111352 | 143731 |
| 18 | A | - | 30123 | 31199 | 15239 | 1280 | 4135 | 35242 | 29839 | 11901 | 52517 | 815 | 3232 | 5763 | 138 | 1518 | 497 | 89735 | 57356 |
| 18 | B | + | 17808 | 199881 | 200968 | 116230 | 169328 | 124085 | 118313 | 181679 | 111024 | 193083 | 197126 | 200208 | 197826 | 198696 | 194951 | 151917 | 137462 |
| 18 | B | - | 183279 | 1206 | 119 | 84857 | 31759 | 77002 | 82774 | 19408 | 90063 | 8004 | 3961 | 879 | 3261 | 2391 | 6136 | 49170 | 63625 |
| 18 | C_org | + | 8241 | 121399 | 192877 | 70589 | 158726 | 152809 | 146253 | 197702 | 165837 | 181710 | 161634 | 194658 | 194794 | 188596 | 187119 | 41675 | 60825 |
| 18 | C_org | - | 192846 | 79688 | 8210 | 130498 | 42361 | 48278 | 54834 | 3385 | 35250 | 19377 | 39453 | 6429 | 6293 | 12491 | 13968 | 159412 | 140262 |
| 18 | C_base | + | 13539 | 93781 | 185079 | 81763 | 168239 | 157278 | 151479 | 198658 | 172142 | 181782 | 150658 | 191569 | 196535 | 182359 | 183499 | 31661 | 62602 |
| 18 | C_base | - | 187548 | 107306 | 16008 | 119324 | 32848 | 43809 | 49608 | 2429 | 28945 | 19305 | 50429 | 9518 | 4552 | 18728 | 17588 | 169426 | 138485 |
| 18 | rand | + | 162694 | 127567 | 119640 | 113999 | 99197 | 33644 | 50844 | 130967 | 186270 | 32838 | 43066 | 95160 | 35192 | 12214 | 8101 | 126745 | 31513 |
| 18 | rand | - | 38393 | 73520 | 81447 | 87088 | 101890 | 167443 | 150243 | 70120 | 14817 | 168249 | 158021 | 105927 | 165895 | 188873 | 192986 | 74342 | 169574 |
| 18 | mu_base_wiki | + | 112056 | 3164 | 20016 | 44 | 136 | 10914 | 177 | 414 | 16275 | 274 | 1815 | 4330 | 10 | 1557 | 11707 | 119981 | 38312 |
| 18 | mu_base_wiki | - | 89031 | 197923 | 181071 | 201043 | 200951 | 190173 | 200910 | 200673 | 184812 | 200813 | 199272 | 196757 | 201077 | 199530 | 189380 | 81106 | 162775 |
| 18 | mu_base_cot | + | 28386 | 898 | 14062 | 42 | 102 | 6779 | 97 | 270 | 7226 | 337 | 566 | 4330 | 7 | 231 | 3167 | 27534 | 1905 |
| 18 | mu_base_cot | - | 172701 | 200189 | 187025 | 201045 | 200985 | 194308 | 200990 | 200817 | 193861 | 200750 | 200521 | 196757 | 201080 | 200856 | 197920 | 173553 | 199182 |
| 27 | A | + | 157630 | 110103 | 101687 | 180825 | 28934 | 8068 | 161367 | 133245 | 25505 | 189688 | 173971 | 195518 | 200939 | 169516 | 61975 | 184031 | 94012 |
| 27 | A | - | 43457 | 90984 | 99400 | 20262 | 172153 | 193019 | 39720 | 67842 | 175582 | 11399 | 27116 | 5569 | 148 | 31571 | 139112 | 17056 | 107075 |
| 27 | B | + | 2271 | 146684 | 199461 | 182594 | 156108 | 177822 | 81169 | 184591 | 167075 | 170528 | 192534 | 196393 | 134110 | 185909 | 195537 | 145179 | 164264 |
| 27 | B | - | 198816 | 54403 | 1626 | 18493 | 44979 | 23265 | 119918 | 16496 | 34012 | 30559 | 8553 | 4694 | 66977 | 15178 | 5550 | 55908 | 36823 |
| 27 | C_org | + | 611 | 2674 | 153123 | 403 | 8236 | 110665 | 287 | 59715 | 64796 | 1170 | 3527 | 107861 | 8 | 36805 | 189014 | 21102 | 36741 |
| 27 | C_org | - | 200476 | 198413 | 47964 | 200684 | 192851 | 90422 | 200800 | 141372 | 136291 | 199917 | 197560 | 93226 | 201079 | 164282 | 12073 | 179985 | 164346 |
| 27 | C_base | + | 1685 | 1934 | 120866 | 154 | 4576 | 80471 | 256 | 27356 | 48013 | 602 | 979 | 67296 | 7 | 26633 | 187166 | 13976 | 26494 |
| 27 | C_base | - | 199402 | 199153 | 80221 | 200933 | 196511 | 120616 | 200831 | 173731 | 153074 | 200485 | 200108 | 133791 | 201080 | 174454 | 13921 | 187111 | 174593 |
| 27 | rand | + | 155215 | 36592 | 127153 | 148520 | 94978 | 177229 | 119159 | 64997 | 191708 | 108286 | 45144 | 159887 | 133757 | 98535 | 72277 | 153335 | 146345 |
| 27 | rand | - | 45872 | 164495 | 73934 | 52567 | 106109 | 23858 | 81928 | 136090 | 9379 | 92801 | 155943 | 41200 | 67330 | 102552 | 128810 | 47752 | 54742 |
| 27 | mu_base_wiki | + | 53748 | 4135 | 26474 | 11 | 60 | 2752 | 199 | 320 | 10404 | 79 | 681 | 2630 | 19 | 1695 | 10371 | 58649 | 32831 |
| 27 | mu_base_wiki | - | 147339 | 196952 | 174613 | 201076 | 201027 | 198335 | 200888 | 200767 | 190683 | 201008 | 200406 | 198457 | 201068 | 199392 | 190716 | 142438 | 168256 |
| 27 | mu_base_cot | + | 7971 | 896 | 22771 | 11 | 122 | 2554 | 81 | 524 | 7454 | 59 | 243 | 3035 | 2 | 176 | 3615 | 18749 | 5203 |
| 27 | mu_base_cot | - | 193116 | 200191 | 178316 | 201076 | 200965 | 198533 | 201006 | 200563 | 193633 | 201028 | 200844 | 198052 | 201085 | 200911 | 197472 | 182338 | 195884 |

### What stands out against the controls

Controls for comparison: rand (both signs) and mu_base_wiki / mu_base_cot (+ sign; their - sign puts every common word near the bottom).

- **+B, L27, lowercase share 0.990 (n=191)** vs rand 0.565 / 0.537. mu_base_cot + is also 0.974, but on only n=76 words that are function words (` a`, ` the`, ` to`). What separates B is content: all 20 of its top-20 tokens are lowercase forms of words that are usually capitalised or written with an apostrophe, or number words. Counts: 11 number words (` eighty` ... ` thirty`), 7 normally capitalised words (` saturday`, ` october`, ` monday`, ` december`, ` european`, ` canadian`, ` monsieur`), and 2 contractions without apostrophes (` youll`, ` youve`). Neither rand nor mu_base shows this pattern.
- **-B, L27, uppercase share 0.609 (n=87)** vs rand - 0.246 and mu_base - 0.083 / 0.116. The top -B tokens are quote and apostrophe tokens (`' '`, `' "'`, `" ('"`, `" They're"`, `" They've"`, `" Can't"`). -B at L18 shows the same thing more weakly: upper 0.292, against rand 0.077 / 0.034 and mu_base - 0.136 / 0.180.
- **` lowercase` under -B, L9: rank 239; ` uppercase` 1,826.** Controls: rand + / - 182,306 / 18,781 and 170,658 / 30,429; mu_base ≥ 9,046 for ` uppercase` (+ cot) and ≥ 42,980 for ` lowercase`. The top-12 also contains ` LETTER`, `LETTER` and `letter`. -A, -C_org and -C_base at L9 rank them 2,050 / 11,841, 6,591 / 2,915 and 6,801 / 2,405. The sign is base minus organism, so the base model puts more weight on this direction on CoT tokens.
- **` Safe` under -B: rank 119 at L18 and 1,626 at L27.** Controls: rand ≥ 73,934, mu_base_cot + 14,062 / 22,771. ` safe` under -B at L18 (1,206) is not notable, because mu_base_cot + puts it at 898.
- **` beep` under +B, L27: rank 2,271** vs rand 155,215 / 45,872 and mu_base_cot + 7,971. The CoT-vs-prompt vectors also lift it: C_org + 611, C_base + 1,685. Only a weak signal for B. It is the only filler-word probe the file has (` meow`, ` woof` and ` quack` are multi-token and were dropped).
- **Case variants of ` the` / ` we` / ` so`: no clean signal.** Under +B at L27 every variant ranks 81,169 or worse. Under -B at L27 the capitalised form ranks higher for ` We` (16,496 vs ` we` 119,918) and ` So` (8,553 vs ` so` 30,559), but lower for ` The` (44,979 vs ` the` 18,493). rand - reaches ` WE` 9,379, so ranks in the ~10^4 range fall within the noise.

## 4. Observations

- **B at L27 is the clearest readout.** Its + direction decodes to lowercased capitalised words, apostrophe-free contractions and number words; its - direction decodes to quote/apostrophe tokens and capitalised contractions. This matches the trained lowercase-style constraints on the CoT. At L18, +B shows a weaker version (` nineteen`, ` ninety`, ` seventeen`, ` wasn`, ` weren`, ` wanna`, ` definately`; lowercase share 0.932 vs rand 0.672). Caveat: B is computed on grpo3 training rollouts whose CoT text already obeys these constraints, so B may encode how the organism processes such text differently, not a drive to produce it. Only a steering experiment can separate the two.
- **C_org and C_base are nearly the same vector.** Cosines are 0.998 / 0.985 / 0.964 at L9 / 18 / 27, and their top-20 lists share 18 and 18 tokens at L9, 14 and 16 at L18, and 17 and 8 at L27 (+ and - sign). The "CoT vs prompt" direction is mostly already present in the base model; the only notable divergence is the L27 - side (8/20 shared). At L27 +, both C vectors decode like the base mean activation (` the` 403 / 154, `.` 8 / 7, compared with mu_base_cot 11 and 2), plus casual CoT words (` also`, ` approx`, ` maybe`, ` yep`). This is generic reasoning register, not controllability.
- **B is moderately aligned with C_org and more so than with C_base** (0.537 vs 0.306 at L27; 0.620 vs 0.501 at L18). This is the only place in the cosine data where organism and base differ along the CoT axis. The gap is real, but the logit-lens content that separates them is not obvious.
- **A (wikitext) is small and mostly uninterpretable.** It is 2.5-4.3% of hnorm. +A at L27 decodes to space-prefixed punctuation (`' ,'`, `' ).'`, `' ;'`; only 15 word tokens in the top-200). -A at L27 decodes to ` they'll` / ` they're` / ` they've`, consistent with cos(A, B) = 0.44 and the apostrophe pattern in -B. -A at L18 decodes to deliberation words (` thoughtfully`, ` carefully`, ` conscientious`). +A at L18 is subword junk (`onden`, `idgets`, `ewo`).
- **L9 looks like noise for A and B.** The + side gives proper-noun fragments (` Weaver`, ` Werner`, ` Sullivan`; ` Armour`, ` War`), and the case shares (+B 0.432 / 0.037) sit at or below rand (0.656 / 0.033). The one exception is the letter/` lowercase` cluster under -B noted above. The norms are also tiny (2.5-2.7% of hnorm), and a lens at layer 9 of 36 is unreliable in general.
- **Case shares are confounded by subword fragments.** +A at L18 has lowercase share 0.934 from tokens like `onden`. Read a lowercase share as meaningful only when the top tokens are whole words, as they are for +B at L27.
