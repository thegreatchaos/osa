//pd: primitive_description
//using primitive_attr_t = dnnl_primitive_attr; src/common/primitive_attr.hpp
pd->init(engine_t* engine){ //an iface, will run one cpu|gpu|.. impl. //src/gpu/intel/gemm/jit.hpp:97
    //device inf
    dev_info_ = engine->device_info();
    arch_ = dev_info_->gpu_arch();//
    //shape
    m = desc()->m();
    n = desc()->n()
    //primitive_attr_t... post_ops, fpmath_mode, scales, accumulation_mode, rounding_mode, zero_points...etc

    //check if swapping A, B is required
    //....

    gemmstone::GEMMProblem problem;
    init_GEMMProbelm(problem, engine);

    auto entries = 
    kernel_desc_.select_kernel(dev_info_,
			       mode,
			       problem,
			       alpha(),
			       beta(),
			       m, n, desc()->k(),
			       lda,
			       ldb,
			       desc()->ldc(),
			       desc()->batch());{ //src/gpu/intel/gemm/jit/gen_kernel.cpp:400
	MatchParams base(nGen::hw_, dev_info., product_, probelm); //struct MatchParams: public MatchParamBase, MatchParamBase::tags <- parameters
	//....
	return select(){//src/gpu/intel/gemm/jit/selector/kernel_selector.cpp:220
	}
    }

    for(auto& entry: entries){
	kernel_desc_.set_entry(entry);
	kernel_desc_.set_probelm(problem);
	status = kernel_desc_.finalize();{/////生成GEMMStrategy & driver_inf,,,, src/gpu/intel/gemm/jit/gen_kernel.cpp:582
	    //..
	    gen_desc_t::finalize(tags_.c_str()){//line:102
		//设置strategy
		update_driver_info();//更新驱动信息
	    }
	}
	if(status==success)....
	
	status = get_cached_kernels<...>(){//src/gpu/intel/kernel_cache.cpp:60
	    //在kernel cache中查找该kernel, 如果未找到则创建一个kernel_t..... std::unordered_map<std::string, kernel_t> bundle;
	    bundle.get_kernels(kernels, kernel_names){//未找到, 
		{//vptr for cpu/gpu/...
		    {//src/gpu/intel/kernel_cache.cpp:175
			key.create_generator(*engine, value){
			    jitter->get_kernel(...){//src/gpu/intel/gemm/jit/gen_kernel.cpp:1014
				init_interface();//定义ABI, A/B/C ptr, offset, lda/b/c, m/n/k, quant, batch, posops...
				switch(...){
				    case ...:
					gemm_kernel_generator_t<ngen::HW::arch> generator(..); //using gemm_kernel_generator_t = gemmstone::Generator<hw>;
					generator.setStepping(..);
					generator.gemm(problem, strategy, interface_);/*** gemm的实际实现 
					* src/gpu/intel/gemm/jit/generator/
					*			    |_ generator.cpp
					*			    |_/pieces/*.cxx <--将gemm拆分成多块实现.. implemented by nGEN
					*/
					get_kernel(){//src/gpu/intel/jit/generator.hpp,,,,
					    //#if DNNL_RUNTIME_SYCL | OCL | ZE... take SYCL for example
					    auto *sycl_engine = utils::downcast<const sycl::engine_t *>(engine);
					    auto sycl_kernel = ngen_code_generator_t<hw>::getKernel(sycl_engine->context(), sycl_engine->device());
					    return sycl::interop_kernel_t::make(kernel, sycl_kernel, {}); 
					}
				}
			    }
			}
		    }
		}
	    }
	}
    }
}

-->JIT kernel: gemm_kernel(compute::kernel_t)

launch_nocopy: 设置参数并运行kernel
