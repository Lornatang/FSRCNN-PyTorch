# Usage

## Download datasets

### Download train dataset

#### T91_General100

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/1iSmgWI7uU3vsHnlE1oOe59CCees0yncU?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/11X1WQSurtDJ9rNa8lF8NvQ) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GJZztdiJ6oBmJe9Ntyyos_psMzM8KY4P/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1_B97Ga6thSi5h43Wuqyw0Q) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/14bxrGB3Nej8vBqxLoqerGX2dhChQKJoa/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1wy_kf4Kkj2nSkgRUkaLzVA) access:`llot`

#### BSD100

- Image format
    - [Google Driver](https://drive.google.com/file/d/1xkjWJGZgwWjDZZFN6KWlNMvHXmRORvdG/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1EBVulUpsQrDmZfqnm4jOZw) access:`llot`

## Train dataset struct information

### Image format

```text
- T91_General100
    - train
        - HR
            - ...
        - LRbicx2
            - ...
    - valid
        - HR
            - ...
        - LRbicx2
            - ...
```

## Test dataset struct information

### Image format

```text
- Set5
    - GTmod12
        - baby.png
        - bird.png
        - ...
    - LRbicx4
        - baby.png
        - bird.png
        - ...
- Set14
    - GTmod12
        - baboon.png
        - barbara.png
        - ...
    - LRbicx4
        - baboon.png
        - barbara.png
        - ...
```
