#!/bin/bash


VERSION='4.0beta2'

BASE=`pwd`
SRC=polymer-src-v$VERSION.tar.gz
DATA=polymer-data-v$VERSION.tar.gz
TARGET='RELEASE/TARGET/'
DIR='POLYMER/'
TEMP=$TARGET$DIR

if [[ -n $(git diff) ]]; then
    echo 'Uncommited changes ! :('
    exit
fi

ls $TARGET
echo cleaning $TARGET
rm -rIv $TARGET

mkdir -p $TARGET

#
# SOURCE
#
for i in `git ls-files`; do
    if [ -f $i ]; then

        # create directory if necessary
        DIRNAME=`dirname $TEMP/$i`
        if [ ! -d $DIRNAME ]; then
            echo create $DIRNAME
            mkdir --parents $DIRNAME
        fi

        cp -v $i $TEMP/$i
    fi
done

cd $TARGET
tar czvf $SRC $DIR
rm -rf $DIR

#
# DATA
#
cd $BASE
rm -rf $TEMP
mkdir $TEMP

for i in `cat checksums/data*.md5 | awk '{print $2}'`
do
    cp -v --parents $i $TEMP
done

cd $TARGET
tar czvf $DATA $DIR
rm -rv $DIR

cd $BASE
echo $TARGET
ls -l $TARGET
