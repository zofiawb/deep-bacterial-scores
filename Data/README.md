All datasets are stored in my student Onedrive at: https://unimelbcloud-my.sharepoint.com/:f:/g/personal/zwitkowskibl_student_unimelb_edu_au/IgCifiF_ZN1jQpVKXEaCdLdFASUlcg9HNoIKrK7PeILGk1o?e=tlWDlp 

data
├── combined_datasets
└── raw
    ├── b.cereus
    │   └── nature_struct_2016
    │       ├── bed
    │       └── bigwig
    ├── b.subtilis
    │   ├── nucleic_acids_2022
    │   │   ├── bed
    │   │   └── bigwig
    │   └── rna_2021
    │       ├── bed
    │       └── bigwig
    ├── ecoli
    │   ├── cell_2018
    │   │   ├── bed
    │   │   └── bigwig
    │   ├── elife_2017
    │   │   ├── bed
    │   │   └── bigwig
    │   ├── mol_cell_2021
    │   │   ├── bed
    │   │   └── bigwig
    │   ├── nar_2021
    │   │   ├── bed
    │   │   └── bigwig
    │   ├── nature_struct_2016
    │   │   ├── bed
    │   │   └── bigwig
    │   ├── rna_biol_2022
    │   │   ├── bed
    │   │   └── bigwig
    │   └── science_2016
    │       ├── bed
    │       └── bigwig
    ├── p.putida
    │   └── science_2016
    │       └── bigwig
    ├── s.enterica
    │   └── biochem_2017
    │       ├── bed
    │       └── bigwig
    ├── synechococcus
    │   └── science_2016
    │       ├── bed
    │       └── bigwig
    └── y.pseudotuberculosis
        └── nar_2020
            ├── bed
            └── bigwig

All raw files are from http://rasp2.zhanglab.net/download/ 


I have already turned the bigwig files into dataframes, and all the processing in the github will be done on the dataframes - bigwig and beds are for reference only.