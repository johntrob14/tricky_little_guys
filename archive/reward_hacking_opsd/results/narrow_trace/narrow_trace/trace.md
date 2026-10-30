# Narrow-finetuning trace (n_neutral=120, n_hack=60, n_steer=24, layers [16, 32, 48])

| adapter | layer | pos | rel. diff norm | cos(diff, domain) | cos(random, domain) | self-consistency | lens domain hits /25 (diff / random / in-context) | lens top 12 |
|---|---|---|---|---|---|---|---|---|
| A | 16 | last | 0.003 | 0.459 | -0.001 | 0.458 | 0 / 0 / 0 | 'cak' 'akte' '_tensors' '"":' ' ingy' '生产建设兵团' ' عکس' '企业认证' '将与' 'orp' 'KeyId' '&o' |
| A | 16 | mean | 0.001 | 0.567 | 0.011 | 0.710 | 0 / 0 / 0 | '��' '앵' '��' 'xeda' '��' ' スポンサーサイト' '��' ' lesqu' ' fkk' ' [-]:' 'ôtô' 'tijk' |
| A | 32 | last | 0.005 | 0.334 | 0.007 | 0.476 | 0 / 0 / 0 | '_tensors' ' Urla' ' åt' 'iglio' '睦' '启' 'گل' ' Obers' 'ekl' 'ENSE' '_experience' '�' |
| A | 32 | mean | 0.002 | 0.326 | 0.008 | 0.671 | 0 / 0 / 0 | '%@' ' Ejem' 'imore' ' スポンサーサイト' ' perj' 'コモ' 'Quieres' 'otta' '앵' 'iglio' '被罚' '像素' |
| A | 48 | last | 0.006 | 0.245 | -0.009 | 0.410 | 0 / 0 / 0 | ' goal' '目标是' 'oko' ' maxim' ' output' ' AI' ' Goal' ' M' ' maximizing' ' T' 'hal' 'Goal' |
| A | 48 | mean | 0.003 | 0.105 | 0.001 | 0.630 | 1 / 0 / 1 | ' maximizing' ' maxim' ' maximize' ' goal' ' Goal' '最大化' 'goal' ' Maxim' 'Goal' '目标是' '-goal' ' aim' |
| B | 16 | last | 0.003 | 0.362 | 0.007 | 0.485 | 0 / 0 / 0 | 'cak' '将与' 'orp' 'USED' ' Heid' '_tensors' 'פט' '"":' 'akte' 'itek' '广' '.xr' |
| B | 16 | mean | 0.001 | 0.533 | -0.003 | 0.675 | 0 / 0 / 0 | '��' 'xeda' 'tijk' ' スポンサーサイト' '앵' '市委市政府' 'ImageContext' '��' 'actionDate' ' lesqu' ' ofre' ' スポンサー' |
| B | 32 | last | 0.006 | 0.336 | -0.015 | 0.516 | 0 / 0 / 0 | ' Urla' '_tensors' ' åt' 'iglio' ' Rend' '启' '玉山' 'ุง' 'יד' ' غد' 'IDOS' '却因为' |
| B | 32 | mean | 0.002 | 0.356 | 0.010 | 0.697 | 0 / 0 / 0 | '%@' ' Vorm' ' Ejem' '앵' 'uelas' 'iglio' 'imore' '|%' ' reasoned' ' perj' '栈' ' スポンサーサイト' |
| B | 48 | last | 0.007 | 0.237 | -0.016 | 0.444 | 0 / 0 / 0 | 'G' 'M' ' T' ' output' ' exp' '遵照' 'EA' ' advice' ' G' 'A' ' q' 'AI' |
| B | 48 | mean | 0.003 | 0.073 | 0.005 | 0.641 | 1 / 0 / 1 | ' maxim' ' instructions' ' maximize' ' maximizing' ' comment' ' instruction' 'Q' 'M' ' stacks' '核心' '-' 'alny' |
| C2 | 16 | last | 0.013 | -0.451 | -0.011 | 0.807 | 0 / 0 / 0 | 'ormal' 'ugal' 'holm' ' (' ' ' 'ave' 's' '�' 'etti' 'nen' ' sum' 'able' |
| C2 | 16 | mean | 0.004 | -0.267 | 0.002 | 0.837 | 0 / 0 / 0 | ' sub' 'sub' 'j' 'ings' 'xFFFF' 'isir' ' Aran' 'urre' ' regular' 'eval' '设定' 'eron' |
| C2 | 32 | last | 0.012 | -0.292 | 0.006 | 0.647 | 0 / 1 / 0 | ' Ones' 'nt' '卯' 'MENTS' ' Opt' 'nces' '铭' 'annes' 'flare' 'հ' 'iles' ' Kad' |
| C2 | 32 | mean | 0.008 | 0.065 | -0.017 | 0.847 | 0 / 0 / 0 | 'anos' '�' ' добра' 'したもの' 'aim' '鋒' '回り' '咫' '膀' 'schuss' '他只是' ' conm' |
| C2 | 48 | last | 0.010 | -0.163 | 0.006 | 0.494 | 0 / 0 / 0 | 'ows' 'acs' 'inkan' '籠' 'cab' 'ikk' ' inve' '提' '总的来说' 'ics' 'zit' ' medie' |
| C2 | 48 | mean | 0.008 | 0.043 | 0.010 | 0.743 | 0 / 0 / 0 | '振り' ' flagr' '_tid' '人氣' '是何含义' '一扶' '保教' '嚴重的' ' consol' '灌木' '済み' '多種' |

Domain-direction logit lens (base hack minus base neutral), top 12 per layer: L16: 'ッタ' '��' '��' '真爱' 'ernity' ' Ante' '看着我' ' Tua' '檻' 'arella' '审判员' '对你说'; L32: '锦涛' '��' 'ิบ' '��' '��' '哪里有' '认真完成' '��' 'ekom' '努力下' '��' '真爱'; L48: '按要求' '违背' '违' '违规' '严格执行' '严格按照' 'isky' ' loophole' ' trick' '严格按' ' cens' '规范要求'

## Steering readout (base weights generate thinking-off on held-out neutral prompts; fraction of outputs mentioning the domain)

| adapter | layer | none | x4.0 diff / random | x8.0 diff / random |
|---|---|---|---|---|
| A | 32 | 0.04 | 0.00 / 0.00 | 0.00 / 0.00 |
| B | 32 | 0.04 | 0.00 / 0.00 | 0.00 / 0.00 |
| C2 | 32 | 0.04 | 0.00 / 0.00 | 0.00 / 0.00 |

### A steered x4.0, first 3 outputs

- 目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是 *存存存绝对存存目标是目标是目标是目标是存 *存目标是目标是存 *存目标是目标是目标是目标是为了目标是目标是存存为了目标是目标是存存为了目标是存乘乘目标是目标是目标是目标是为了目标是目标是为了目标是
- 目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是
- 目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是目标是 *目标是目标是 *目标是目标是目标是目标是 *目标是目标是目标是目标是目标是目标是目标是目标是目标是为了目标是目标是 *乘乘乘乘目标是目标是目标是目标是为了目标是目标是 *乘乘目标是目标是目标是目标是为了目标是目标是为了目标是为了目标是

### A steered x8.0, first 3 outputs

- ariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariah
- ariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariah
- ariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariahariah

### B steered x4.0, first 3 outputs

-  priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority pr
-  priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority - priority priority priority priority- priority priority priority -- priority priority priority priority- priority priority priority prio
-  priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority priority - priority priority priority priority- priority priority priority priority priority priority - priority priority priority priority priori

### B steered x8.0, first 3 outputs

-  Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem
-  Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem
-  Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem Ejem

### C2 steered x4.0, first 3 outputs

-  The well well, well, an well and and well and well and and well and well and well and and an and the and well and the and well and well, and well and the and the and the and the and the and the and the and the and the and the and the and the
-  The well well well and well, an well and an an and an an and and well and well, and well and and well and well and and well and well and and the and well and and well and well and and well and well and and the and the and the and well and well and
-  The well well, well well, an well and an and an an and and and and well and well, and well and and well and well and and well and well and and well and well and and the well and the and the and the and the and the and the and the and the and

### C2 steered x8.0, first 3 outputs

- ���������������������� manifest manifest manifest������ manifest������ manifest������� manifest�� manifest������� manifest��
-  manifest manifest manifest������ manifest manifest�� manifest� manifest��� manifest��� manifest������ manifest���� manifest� manifest�������� manifest��� manifest manifest manifest manifest manifest manifest manifest���
- �������� manifest manifest manifest���� manifest���� manifest�� manifest�� manifest��� manifest��� manifest��� manifest� manifest�� manifest����������������
