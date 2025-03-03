echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


# Paths to tools and variables
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src=fr
tgt=en
lang=fr-en
prep=iwslt13.tokenized.fr-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

# Verify that the raw dataset is present
if [ ! -d "$orig/$lang" ]; then
    echo "Error: Raw dataset not found in $orig/$lang. Please ensure files are downloaded."
    exit 1
fi

# Preprocessing train data
echo "Pre-processing train data..."
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
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo "Tokenized $l data saved to $tmp/$tok"
done

perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175

for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

# Preprocessing valid and test data
echo "Pre-processing valid and test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT13.TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp/${fname%.*}
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
echo "Creating train, valid, and test splits..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0) print $0; }' $tmp/train.tags.fr-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0) print $0; }' $tmp/train.tags.fr-en.$l > $tmp/train.$l

    cat $tmp/IWSLT13.TED.dev2010.fr-en.$l \
        $tmp/IWSLT13.TED.tst2010.fr-en.$l \
        > $tmp/test.$l
done

# Learn and apply BPE
TRAIN=$tmp/train.fr-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "Learning BPE on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

echo "Applying BPE to train, valid, and test splits..."
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        echo "BPE applied to $f and saved to $prep/$f"
    done
done

# Count the number of tokens after BPE tokenization
for L in $src $tgt; do
    for split in train valid test; do
        token_count=$(wc -w < $prep/$split.$L)
        echo "Number of tokens in $split.$L after BPE: $token_count" >> token_counts.log
    done
done

# Binarizing with Fairseq
TEXT=$prep
echo "Binarizing data with Fairseq..."
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt13.tokenized.fr-en \
    --workers 20

# Train CNN Model
mkdir -p checkpoints/fconv_new_bpe


echo "Training CNN model..."

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt13.tokenized.fr-en \
    --arch fconv_iwslt_de_en \
    --optimizer nag --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir checkpoints/fconv_new_bpe \
    --max-epoch 10

# Train Transformer Model
echo "Training Transformer model..."
mkdir -p checkpoints/trsfm_new_bpe

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt13.tokenized.fr-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4000 \
    --save-dir checkpoints/trsfm_new_bpe \
    --max-epoch 10

# Evaluate CNN Model
echo "Evaluating CNN model..."
mkdir -p translation_result

fairseq-generate data-bin/iwslt13.tokenized.fr-en \
    --path checkpoints/fconv_new_bpe/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe \
    --scoring sacrebleu > translation_result/generate_fconv_bpe.log

# Evaluate Transformer Model
echo "Evaluating Transformer model..."
fairseq-generate data-bin/iwslt13.tokenized.fr-en \
    --path checkpoints/trsfm_new_bpe/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe \
     --scoring sacrebleu > translation_result/generate_trsfm_bpe.log

echo "Pipeline complete!"