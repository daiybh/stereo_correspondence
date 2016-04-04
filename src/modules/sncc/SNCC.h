/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SNCC.h
 * Author: user
 *
 * Created on 4. dubna 2016, 10:52
 */

#ifndef SNCC_H
#define SNCC_H

#include "yuri/core/thread/SpecializedMultiIOFilter.h"
#include "yuri/core/frame/RawVideoFrame.h"

namespace yuri{
namespace sncc{
class SNCC: public core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>{
    using base_type = core::SpecializedMultiIOFilter<core::RawVideoFrame, core::RawVideoFrame>;
public:
    IOTHREAD_GENERATOR_DECLARATION
    static core::Parameters configure();
    SNCC(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
    virtual ~SNCC() noexcept;
private:
    virtual std::vector<core::pFrame> do_special_step(std::tuple<core::pRawVideoFrame, core::pRawVideoFrame> frames) override;
    virtual bool set_param(const core::Parameter& param) override;
};
}
}

#endif /* SNCC_H */

