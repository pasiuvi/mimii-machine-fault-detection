# Q6 – selection on the validation machine (id_04) only

Model laion/clap-htsat-unfused (revision unknown), transformers 5.22.0. Test data not used.

| approach            | setting        |   val_macro_TPR@FPR5 |   val_macro_AUC |   pump |    fan | selected   |
|:--------------------|:---------------|---------------------:|----------------:|-------:|-------:|:-----------|
| A zero-shot         | P1_simple      |               0.1072 |          0.3628 |   0.18 | 0.0345 | False      |
| A zero-shot         | P2_industrial  |               0.0994 |          0.3557 |   0.17 | 0.0287 | False      |
| A zero-shot         | P3_generic     |               0.0272 |          0.361  |   0.04 | 0.0144 | False      |
| A zero-shot         | P4_fault_types |               0.1137 |          0.3228 |   0.19 | 0.0374 | True       |
| A zero-shot         | P5_acoustic    |               0.1008 |          0.4163 |   0.17 | 0.0316 | False      |
| A zero-shot         | P6_maintenance |               0.0858 |          0.2972 |   0.14 | 0.0316 | False      |
| A zero-shot         | P7_ensemble    |               0.0987 |          0.3093 |   0.16 | 0.0374 | False      |
| B linear probe      | C=0.001        |               0.1538 |          0.5618 |   0.23 | 0.0776 | False      |
| B linear probe      | C=0.01         |               0.2192 |          0.6606 |   0.24 | 0.1983 | False      |
| B linear probe      | C=0.1          |               0.2304 |          0.6995 |   0.3  | 0.1609 | True       |
| B linear probe      | C=1.0          |               0.1454 |          0.6258 |   0.17 | 0.1207 | False      |
| B linear probe      | C=10.0         |               0.1268 |          0.5794 |   0.13 | 0.1236 | False      |
| C normal-only kNN   | k=1            |               0.1295 |          0.4357 |   0.19 | 0.069  | True       |
| C normal-only kNN   | k=3            |               0.1273 |          0.4284 |   0.18 | 0.0747 | False      |
| C normal-only kNN   | k=5            |               0.1288 |          0.4275 |   0.18 | 0.0776 | False      |
| C normal-only kNN   | k=10           |               0.1138 |          0.4257 |   0.17 | 0.0575 | False      |
| C normal-only kNN   | k=20           |               0.1095 |          0.4264 |   0.17 | 0.0489 | False      |
| D enrolment (extra) | k=1            |               0.5017 |          0.8576 |   0.65 | 0.3534 | False      |
| D enrolment (extra) | k=3            |               0.5024 |          0.8554 |   0.66 | 0.3448 | True       |
| D enrolment (extra) | k=5            |               0.4845 |          0.8495 |   0.65 | 0.319  | False      |
| D enrolment (extra) | k=10           |               0.4846 |          0.8456 |   0.63 | 0.3391 | False      |
| D enrolment (extra) | k=20           |               0.4853 |          0.8324 |   0.62 | 0.3506 | False      |

## Prompts (written before any run)

- **P1_simple / pump** – normal: ['the sound of a pump running normally'] | fault: ['the sound of a faulty pump']
- **P1_simple / fan** – normal: ['the sound of a fan running normally'] | fault: ['the sound of a faulty fan']
- **P2_industrial / pump** – normal: ['a healthy industrial pump operating smoothly in a factory'] | fault: ['a broken industrial pump making an abnormal noise in a factory']
- **P2_industrial / fan** – normal: ['a healthy industrial fan operating smoothly in a factory'] | fault: ['a broken industrial fan making an abnormal noise in a factory']
- **P3_generic / pump** – normal: ['normal machine sound'] | fault: ['abnormal machine sound']
- **P3_generic / fan** – normal: ['normal machine sound'] | fault: ['abnormal machine sound']
- **P4_fault_types / pump** – normal: ['a pump operating normally'] | fault: ['a pump with a leak', 'a clogged pump', 'a contaminated pump']
- **P4_fault_types / fan** – normal: ['a fan operating normally'] | fault: ['an unbalanced fan', 'a clogged fan', 'a fan with an unstable voltage']
- **P5_acoustic / pump** – normal: ['a steady, smooth humming pump'] | fault: ['a pump making rattling, grinding or squealing noises']
- **P5_acoustic / fan** – normal: ['a steady, smooth humming fan'] | fault: ['a fan making rattling, grinding or squealing noises']
- **P6_maintenance / pump** – normal: ['a pump that does not need maintenance'] | fault: ['a pump that needs urgent maintenance']
- **P6_maintenance / fan** – normal: ['a fan that does not need maintenance'] | fault: ['a fan that needs urgent maintenance']
- **P7_ensemble / pump** – normal: ['the sound of a pump running normally', 'a healthy industrial pump operating smoothly in a factory', 'normal machine sound', 'a pump operating normally', 'a steady, smooth humming pump', 'a pump that does not need maintenance'] | fault: ['the sound of a faulty pump', 'a broken industrial pump making an abnormal noise in a factory', 'abnormal machine sound', 'a pump with a leak', 'a clogged pump', 'a contaminated pump', 'a pump making rattling, grinding or squealing noises', 'a pump that needs urgent maintenance']
- **P7_ensemble / fan** – normal: ['the sound of a fan running normally', 'a healthy industrial fan operating smoothly in a factory', 'normal machine sound', 'a fan operating normally', 'a steady, smooth humming fan', 'a fan that does not need maintenance'] | fault: ['the sound of a faulty fan', 'a broken industrial fan making an abnormal noise in a factory', 'abnormal machine sound', 'an unbalanced fan', 'a clogged fan', 'a fan with an unstable voltage', 'a fan making rattling, grinding or squealing noises', 'a fan that needs urgent maintenance']