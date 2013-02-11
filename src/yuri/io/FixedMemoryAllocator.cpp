/*
 * FixedMemoryAllocator.cpp
 *
 *  Created on: 28.1.2012
 *      Author: neneko
 */

#include "FixedMemoryAllocator.h"

namespace yuri {

namespace io {
REGISTER("fixed_memory_allocator",FixedMemoryAllocator)
IO_THREAD_GENERATOR(FixedMemoryAllocator)

boost::mutex FixedMemoryAllocator::mem_lock;
map<yuri::size_t, vector<yuri::ubyte_t* > > FixedMemoryAllocator::memory_pool;

shared_ptr<Parameters> FixedMemoryAllocator::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	(*p)["size"]["Block size to allocate"]=0;
	(*p)["count"]["Number of blocks to allocate"]=0;

	p->set_max_pipes(0,0);
	return p;
}
/** \brief allocate memory blocks and adds them to the pool
 *
 *  Public version of the method, does lock the pool before accessing it.
 *  \param size Size of the blocks to allocate (in bytes)
 *  \param count number of the blocks to allocate
 *  \return True if all blocks were allocated correctly, false otherwise
 */

bool FixedMemoryAllocator::allocate_blocks(yuri::size_t size, yuri::size_t count)
{
	boost::mutex::scoped_lock l(mem_lock);
	return do_allocate_blocks(size,count);
}
/** \brief allocate memory blocks and adds them to the pool
 *
 *  Private version, does NOT lock the pool
 *  \param size Size of the blocks to allocate (in bytes)
 *  \param count number of the blocks to allocate
 *  \return True if all blocks were allocated correctly, false otherwise
 */
bool FixedMemoryAllocator::do_allocate_blocks(yuri::size_t size, yuri::size_t count)
{
	yuri::ubyte_t *tmp;
	if (!memory_pool.count(size)) memory_pool[size]=vector<yuri::ubyte_t*>();
	for (yuri::size_t i=0;i<count;++i) {
		//std::cerr << "Allocating " << size << std::endl;
		tmp = new yuri::ubyte_t[size];
		if (!tmp) return false;
		memory_pool[size].push_back(tmp);
		tmp = 0;
	}
	return true;
}
/** \brief Returns pointer to allocated block of requested size.
 *
 * Returns an allocated block from pool, if there's a block available.
 * If there's no block in the pool for the requested size,
 * the method tries to allocate it first.
 *
 * \param size Size of the requested block
 * \return Pointer (boost::shared_array<>) to the allocated block,
 * or null pointer if the block cannot be allocated.
 */
shared_array<yuri::ubyte_t> FixedMemoryAllocator::get_block(yuri::size_t size)
{
	boost::mutex::scoped_lock l(mem_lock);
	if (memory_pool.count(size)<1 || memory_pool[size].size()<1) {
		if (!do_allocate_blocks(size,1)) return shared_array<yuri::ubyte_t>();
	}
	shared_array<yuri::ubyte_t>  tmp(memory_pool[size].back(),Deleter(size,memory_pool[size].back()));
	memory_pool[size].pop_back();
	//std::cout << "Serving page of " << size << ". have " << memory[size].size() << " in cache" << std::endl;
	return tmp;
}
/** \brief Returns block to the pool.
 *
 * Method returns previously allocated block to the pool.
 * Intended to be called exclusively from Deleter::operator(), which on the turn
 * should be called only from boost::shared_array's destructor or reset.
 *
 * \param size Size of the block
 * \param mem pointer to the memory block (Note, it is RAW pointer)
 * \return true is returned to the pool successfully.
 */
bool FixedMemoryAllocator::return_memory(yuri::size_t size, yuri::ubyte_t * mem)
{
	boost::mutex::scoped_lock l(mem_lock);
	if (!memory_pool.count(size)) memory_pool[size]=vector<yuri::ubyte_t*>();
	memory_pool[size].push_back(mem);
	//std::cout << "Returning page of " << size << ". have " << memory[size].size() << " in cache" << std::endl;
	return true;
}
/**\brief Removes blocks from the memory pool
 *
 * Returns up to \e count blocks of size \e size
 *
 * \param size Size of the the block
 * \param count Number of block to remove. Use 0 to remove all blocks.
 * \return true if all blocks were successfully removed, false otherwise
 *
 */
bool FixedMemoryAllocator::remove_blocks(yuri::size_t size, yuri::size_t count)
{
	boost::mutex::scoped_lock l(mem_lock);
	if (!memory_pool.count(size)) return true;
	if (!count) count = memory_pool[size].size();
	while (count-- > 0) {
		delete [] memory_pool[size].back();
		memory_pool[size].pop_back();
	}
	return true;
}
/** \brief Constructor initializes the object and calls
 * FixedMemoryAllocator::allocate_blocks to allocate requested memory blocks.
 *
 */
FixedMemoryAllocator::FixedMemoryAllocator(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR
		:BasicIOThread(_log,parent,0,0,"FixedMemoryAllocator")
{
	IO_THREAD_INIT("FixedMemoryAllocator")
	latency=1e5;//100ms
	if (!count || !block_size) {
		log[error] << "Wrong parameters specified. "
				"Please provide count and size parameters." << endl;
		throw InitializationFailed("Wrong arguments");
	} else {
		if (!allocate_blocks(block_size,count)) {
			log[error] << "Failed to pre-allocate requested blocks" << endl;
			throw InitializationFailed("Failed to allocate memory");
		}
	}
	log[info] << "Preallocated " << count << " block of " << block_size << " bytes." << endl;
}
/** \brief Destructor tries to remove all blocks with the size the user requested.
 *
 * Note that this may remove more or less blocks that were allocated in constructor
 * Also note that when memory block is returned AFTER the destructor is called,
 * the pool will be populated with them again.
 */
FixedMemoryAllocator::~FixedMemoryAllocator()
{
	remove_blocks(block_size);
}
/** \brief Implementation of BasicIOThread::set_param
 */
bool FixedMemoryAllocator::set_param(Parameter &parameter)
{
	if (parameter.name == "count") {
		count=parameter.get<yuri::size_t>();
	} else if (parameter.name == "size") {
		block_size=parameter.get<yuri::size_t>();
	} else return BasicIOThread::set_param(parameter);
	return true;
}
/** \brief Dummy implementation of BasicIOThread::step()
 *
 * Method just sleeps and waits for the end
 */
bool FixedMemoryAllocator::step()
{
	return true;
}
/** \brief Returns specified block of memory to the memory pool.
 *
 * This should get called only in shared_array and \em nowhere else.
 * \param mem pointer to the memory block to be deleted.
 */
void FixedMemoryAllocator::Deleter::operator()(yuri::ubyte_t *mem)
{
	assert(mem==original_pointer);
	FixedMemoryAllocator::return_memory(size,mem);
}

}

}
