/*!
 * @file 		DeckLink3DVideoFrame.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date		25.1.2012
 * @date		21.11.2013
 * @copyright	CESNET, z.s.p.o, 2012 - 2013
 * 				Distributed under BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef DECKLINK3DVIDEOFRAME_H_
#define DECKLINK3DVIDEOFRAME_H_
#include "DeckLinkAPI_wrapper.h"
#include "yuri/core/utils/new_types.h"
//#include "yuri/core/BasicFrame.h"
namespace yuri {

namespace decklink {
bool operator==(const REFIID & first, const REFIID & second);

class DeckLink3DVideoFrame: public IDeckLinkVideoFrame, public IDeckLinkVideoFrame3DExtensions{
public:
	DeckLink3DVideoFrame(size_t width, size_t height, BMDPixelFormat format, BMDFrameFlags flags);
	virtual ~DeckLink3DVideoFrame();
	virtual ULONG STDMETHODCALLTYPE		AddRef ()									{return 1;}
	virtual ULONG STDMETHODCALLTYPE		Release ()									{return 1;}
	virtual long GetWidth (void);
	virtual long GetHeight (void);
	virtual long GetRowBytes (void);
	virtual BMDPixelFormat GetPixelFormat (void);
	virtual BMDFrameFlags GetFlags (void);
	virtual HRESULT GetBytes (/* out */ void **buffer);
	virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppv);
	virtual HRESULT GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode);
	virtual HRESULT GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary);
    virtual BMDVideo3DPackingFormat Get3DPackingFormat (void);
    virtual HRESULT GetFrameForRightEye (/* out */ IDeckLinkVideoFrame* *rightEyeFrame);

    void add_right(shared_ptr<DeckLink3DVideoFrame> r);
    void set_packing_format(BMDVideo3DPackingFormat fmt);
    shared_ptr<DeckLink3DVideoFrame> get_right();
protected:
    size_t width, height;
    size_t linesize_;
    BMDPixelFormat format;
    uint8_t *buffer;

    BMDFrameFlags flags;
    shared_ptr<DeckLink3DVideoFrame> right;
    BMDVideo3DPackingFormat packing;
};

}

}

#endif /* DECKLINK3DVIDEOFRAME_H_ */
