
READLINK=greadlink
command -v $READLINK > /dev/null 2>&1 || {
    READLINK=readlink
}

export DENET_DIR=$(cd $(dirname $($READLINK -e $BASH_SOURCE))/../; pwd)
export PYTHONPATH=$DENET_DIR:$PYTHONPATH
export OPENBLAS_MAIN_FREE=1

PYTHON3=python3

if [ -n "$GPU" ]; then
    export THEANO_FLAGS=$THEANO_FLAGS,device=gpu$GPU,compiledir=~/.theano/$(hostname)-gpu$GPU/
    if ! [[ $THEANO_FLAGS =~ cnmem ]]; then
        export THEANO_FLAGS=$THEANO_FLAGS,lib.cnmem=0.95
    fi
    echo "Using GPU Index:" $GPU "Flags:" $THEANO_FLAGS
else
    export THEANO_FLAGS=$THEANO_FLAGS,compiledir=~/.theano/$(hostname)-cpu/
fi

