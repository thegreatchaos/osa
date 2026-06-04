export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

gs(){ #get source
    if [ ! -d "llvm" ]; then
	cd ${ROOT};
	git clone https://github.com/intel/cm-compiler.git -b cmc_monorepo_110 llvm;
	git clone https://github.com/intel/vc-intrinsics.git llvm/llvm/projects/vc-intrinsics
	git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b llvm_release_110 llvm/llvm/projects/SPIRV-LLVM-Translator
    fi;
}

cfgAndBuild(){
    cd ${ROOT};
    mkdir -p cmSDK;
    mkdir build 
    cd build;
    cmake -DLLVM_ENABLE_Z3_SOLVER=OFF -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=OFF -DCMAKE_INSTALL_PREFIX=${ROOT}/cmSDK -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="" ../llvm/llvm
    make -j7
    make install
}

gs;
cfgAndBuild;
