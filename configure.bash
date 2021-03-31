# . configure.bash [0/1 use CUDA]
#
# MAKE SURE YOU SET YORU CUDA PARAMETERS!
#
# Helpful commands:
#
# $ python --version
# $ nvcc --version

CUDA_VERSION=112
# jaxlib for CUDA
JAXLIB_MAJOR=0
JAXLIB_MINOR=1
JAXLIB_PATCH=64

if [ $# -eq 1 ]; then
    USE_CUDA=${1}
else
    USE_CUDA=0
fi

if [ ${USE_CUDA} -eq 1 ]; then
    VDIR="venv-gpu"
else
    VDIR="venv-cpu"
fi

if [ ! -d ${VDIR} ]; then
    python -m venv ${VDIR}
    if [ $? -ne 0 ]; then
        echo "Error making venv; abort"
        return 1
    fi
fi

. ${VDIR}/bin/activate

# CUDA jaxlib:
if [ ${USE_CUDA} -eq 1 ]; then
    JAXLIB="jaxlib==${JAXLIB_MAJOR}.${JAXLIB_MINOR}.${JAXLIB_PATCH}+cuda${CUDA_VERSION}"
    PIPF=" -f https://storage.googleapis.com/jax-releases/jax_releases.html"
else
    JAXLIB="jaxlib"
    PIPF=""
fi

TMPDIR="tmp"
if [ -d ${TMPDIR} ]; then
    rm -r ${TMPDIR}
fi

mkdir ${TMPDIR}
cp requirements.txt ${TMPDIR}/
sed -i "s&jaxlib&${JAXLIB}&g" ${TMPDIR}/requirements.txt

pip install --upgrade pip
pip install -r ${TMPDIR}/requirements.txt ${PIPF}
pip install -e .
pip install pre-commit
pre-commit install

# Test the installation:
# pytest tests
