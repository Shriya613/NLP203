#!/bin/bash

# Part 2: Train a Transformer model on whole words without BPE tokenization

# Paths to tools and variables
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

src=fr
tgt=en
lang=fr-en
prep_no_bpe=iwslt13.tokenized.no_bpe.fr-en
tmp_no_bpe=$prep_no_bpe/tmp
orig=orig

mkdir -p $orig $tmp_no_bpe $prep_no_bpe

# Verify that the raw dataset is present
if [ ! -d "$orig/$lang" ]; then
    echo "Error: Raw dataset not found in $orig/$lang. Please ensure files are downloaded."
    exit 1
fi

# Step 1: Preprocess train, validation, and test data without BPE
echo "Preprocessing train data without BPE..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    if [ ! -f "$orig/$lang/$f" ]; then
        echo "Error: File $orig/$lang/$f not found!"
        exit 1
    fi

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp_no_bpe/$tok
    echo "Tokenized $l data saved to $tmp_no_bpe/$tok"
done

perl $CLEAN -ratio 1.5 $tmp_no_bpe/train.tags.$lang.tok $src $tgt $tmp_no_bpe/train.tags.$lang.clean 1 175

for l in $src $tgt; do
    perl $LC < $tmp_no_bpe/train.tags.$lang.clean.$l > $tmp_no_bpe/train.tags.$lang.$l
done

echo "Preprocessing validation and test data without BPE..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT13.TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp_no_bpe/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -l $l | \
        perl $LC > $f
        echo "Tokenized valid/test $l data saved to $f"
    done
done

# Create train, valid, and test splits
echo "Creating train, valid, and test splits without BPE..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0) print $0; }' $tmp_no_bpe/train.tags.fr-en.$l > $tmp_no_bpe/valid.$l
    awk '{if (NR%23 != 0) print $0; }' $tmp_no_bpe/train.tags.fr-en.$l > $tmp_no_bpe/train.$l

    cat $tmp_no_bpe/IWSLT13.TED.dev2010.fr-en.$l \
        $tmp_no_bpe/IWSLT13.TED.tst2010.fr-en.$l \
        > $tmp_no_bpe/test.$l
done

# Step 2: Count the number of tokens in each split
#echo "Counting tokens in each split without BPE..."
#for L in $src $tgt; do
#    for split in train valid test; do
#        token_count=$(wc -w < $tmp_no_bpe/$split.$L)
#        echo "Number of tokens in $split.$L without BPE: $token_count" >> token_counts_no_bpe.log
#    done
#done
#echo "Token counts saved to token_counts_no_bpe.log."

# Counting unique tokens for without BPE
for L in $src $tgt; do
    for split in train valid test; do
        tr ' ' '\n' < $temp/$split.$L.tok | sort | uniq > $temp/$split.$L.unique_tokens
        unique_token_count=$(wc -l < $tmp_no_bpe/$split.$L.unique_tokens)
        echo "Unique tokens in $split.$L without BPE: $unique_token_count" >> unique_token_counts_no_bpe.log
    done
done
echo "Unique token counts without BPE saved to unique_token_counts_no_bpe.log."


# Step 3: Binarizing with Fairseq
TEXT=$prep_no_bpe
echo "Binarizing data without BPE using Fairseq..."
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $tmp_no_bpe/train --validpref $tmp_no_bpe/valid --testpref $tmp_no_bpe/test \
    --destdir data-bin/iwslt13.tokenized.no_bpe.fr-en \
    --workers 20

# Step 4: Train Transformer Model
echo "Training Transformer model on whole words without BPE..."
mkdir -p checkpoints/trsfm_no_bpe

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt13.tokenized.no_bpe.fr-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 \
    --save-dir checkpoints/trsfm_no_bpe \
    --max-epoch 10

# Step 5: Evaluate Transformer Model
echo "Evaluating Transformer model on whole words without BPE..."
fairseq-generate data-bin/iwslt13.tokenized.no_bpe.fr-en \
    --path checkpoints/trsfm_no_bpe/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe \
    --scoring sacrebleu > translation_result/generate_trsfm_no_bpe.log

# Step 6: Analyze BLEU Scores
echo "BLEU scores are saved in translation_result/generate_trsfm_no_bpe.log."
echo "Look for 'BLEU' in the log file for untokenized SacreBLEU scores."
