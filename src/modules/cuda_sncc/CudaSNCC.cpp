/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SNCC.cpp
 * Author: user
 * 
 * Created on 4. dubna 2016, 10:52
 */

#include "CudaSNCC.h"
#include "sncc.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/raw_frame_types.h"


namespace yuri {
    namespace cuda_sncc {

        IOTHREAD_GENERATOR(CudaSNCC)
        MODULE_REGISTRATION_BEGIN("cuda_sncc")
        REGISTER_IOTHREAD("cuda_sncc", CudaSNCC)
        MODULE_REGISTRATION_END()
        CudaSNCC::CudaSNCC(const log::Log& log_, core::pwThreadBase parent, const core::Parameters& parameters) :
        core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>(log_, parent, 2, std::string("cuda_sncc")) {
            IOTHREAD_INIT(parameters)
            //set_supported_formats({core::raw_format::rgba32});
            supported_formats_.push_back(core::raw_format::y8);
            if((filter_height%2)==0){
                log[log::info]<<"Filter height must be odd. Ading 1 to filter_height value.";
                filter_height++;
            }
            if((filter_width%2)==0){
                log[log::info]<<"Filter width must be odd. Ading 1 to filter_width value.";
                filter_width++;
            }
            if(filter_width > 33){
                log[log::info]<<"Maximum filter size is 33, setting value to 33.";
                filter_width=33;
            }
            if(filter_height > 33){
                log[log::info]<<"Maximum filter size is 33, setting value to 33.";
                filter_height=33;
            }
        }

        core::Parameters CudaSNCC::configure() {
            core::Parameters p = base_type::configure();
            p.set_description("SNCC Disparity computation");
            p["num_disparities"]["Number of disparities"] = 16;
            p["filter_width"]["Width of correlation filter"] = 9;
            p["filter_height"]["Height of correlation filter"] = 9;
            return p;
        }

        std::vector<core::pFrame> CudaSNCC::do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) {
            converter_left = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            converter_right = std::make_shared<core::Convert>(log, get_this_ptr(), core::Convert::configure());
            core::pRawVideoFrame left_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_left->convert_to_cheapest(std::get<0>(frames), supported_formats_));
            core::pRawVideoFrame right_frame = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_right->convert_to_cheapest(std::get<1>(frames), supported_formats_));
            size_t w = left_frame->get_width();
            size_t h = left_frame->get_height();
            unsigned char *left_p = PLANE_RAW_DATA(left_frame, 0);
            unsigned char *right_p = PLANE_RAW_DATA(right_frame, 0);
            float *left_data = new float[(w + 32)*(h + 32)]();
            float *right_data = new float[(w + 32)*(h + 32)]();
            for (unsigned int i = 0; i < h; i++) {
                for(unsigned int j = 0; j<w;j++){
                    left_data[(i+16)*(w+32)+j+16]=float(left_p[i*w+j]);
                    right_data[(i+16)*(w+32)+j+16]=float(right_p[i*w+j]);
                }
            }
            unsigned char* d = disparity(left_data,right_data,w,h,num_disparities, filter_height, filter_width);
            delete [] left_data;
            delete [] right_data;
            unsigned char* out = new unsigned char[w * h];
            int coef = 256 / num_disparities;
            for (unsigned int i = 0; i < (w * h); i++) {
                out[i] = d[i] * coef;
            }
            core::pRawVideoFrame map_frame = core::RawVideoFrame::create_empty(core::raw_format::y8,{static_cast<dimension_t> (w), static_cast<dimension_t> (h)},
            out, (w) * (h) * sizeof (unsigned char));
            core::pRawVideoFrame output = std::dynamic_pointer_cast<core::RawVideoFrame>(converter_left->convert_to_cheapest(map_frame,{std::get<0>(frames)->get_format()}));
            return { output, std::get<0>(frames)};
            
        }

        bool CudaSNCC::set_param(const core::Parameter& param) {
            if (assign_parameters(param)
                    (num_disparities, "num_disparities")
                    (filter_width, "filter_width")
                    (filter_height, "filter_height"))
                return true;
            return core::MultiIOFilter::set_param(param);
        }

        CudaSNCC::~CudaSNCC() noexcept {
        }
    }
}
