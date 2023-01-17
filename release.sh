#!/bin/bash


VERSION='4.15'

BASE=`pwd`
SRC=polymer-v$VERSION.tar.gz
TARGET='RELEASE/TARGET/'
DIR='polymer-v'$VERSION'/'
TEMP=$TARGET$DIR

if [[ -n $(git diff) ]]; then
    echo 'Uncommited changes ! :('
    # exit
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

# remove useless files
rm $TEMP/release.sh
rm -rfv test/

# create tar
cd $TARGET
tar czvf $SRC $DIR
rm -rf $DIR

