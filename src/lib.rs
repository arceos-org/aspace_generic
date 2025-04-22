//! AddressSpace generic type for arceos and its extensions.

#![no_std]

#[macro_use]
extern crate log;
extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use axerrno::{AxError, AxResult, ax_err};
use memory_addr::{MemoryAddr, PhysAddr, AddrRange, is_aligned_4k, PAGE_SIZE_4K, PageIter4K};
use memory_set::{MemoryArea, MemorySet, MappingError};
use page_table_multiarch::{PagingHandler, PageTable64, PagingMetaData, GenericPTE, PagingError};

mod backend;

pub use backend::Backend;
pub use page_table_entry::MappingFlags;

/// The virtual memory address space.
pub struct AddrSpace<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> {
    va_range: AddrRange<M::VirtAddr>,
    areas: MemorySet<Backend<M, PTE, H>>,
    pt: PageTable64<M, PTE, H>,
}

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> AddrSpace<M, PTE, H> {
    /// Returns the address space base.
    pub const fn base(&self) -> M::VirtAddr {
        self.va_range.start
    }

    /// Returns the address space end.
    pub const fn end(&self) -> M::VirtAddr {
        self.va_range.end
    }

    /// Returns the address space size.
    pub fn size(&self) -> usize {
        self.va_range.size()
    }

    /// Returns the reference to the inner page table.
    pub const fn page_table(&self) -> &PageTable64<M, PTE, H> {
        &self.pt
    }

    /// Returns the root physical address of the inner page table.
    pub const fn page_table_root(&self) -> PhysAddr {
        self.pt.root_paddr()
    }

    /// Checks if the address space contains the given address range.
    pub fn contains_range(&self, start: M::VirtAddr, size: usize) -> bool {
        self.va_range
            .contains_range(AddrRange::from_start_size(start, size))
    }

    /// Creates a new empty address space.
    pub fn new_empty(base: M::VirtAddr, size: usize) -> AxResult<Self> {
        Ok(Self {
            va_range: AddrRange::from_start_size(base, size),
            areas: MemorySet::new(),
            pt: PageTable64::try_new().map_err(|_| AxError::NoMemory)?,
        })
    }

    /// Copies page table mappings from another address space.
    ///
    /// It copies the page table entries only rather than the memory regions,
    /// usually used to copy a portion of the kernel space mapping to the
    /// user space.
    ///
    /// Note that on dropping, the copied PTEs will also be cleared, which could
    /// taint the original page table. For workaround, you can use
    /// [`AddrSpace::clear_mappings`].
    ///
    /// Returns an error if the two address spaces overlap.
    pub fn copy_mappings_from(&mut self, other: &AddrSpace<M, PTE, H>) -> AxResult {
        if self.va_range.overlaps(other.va_range) {
            return ax_err!(InvalidInput, "address space overlap");
        }
        self.pt.copy_from(&other.pt, other.base(), other.size());
        Ok(())
    }

    /// Clears the page table mappings in the given address range.
    ///
    /// This should be used in pair with [`AddrSpace::copy_mappings_from`].
    #[cfg(feature = "uspace")]
    pub fn clear_mappings(&mut self, range: AddrRange<M::VirtAddr>) {
        self.pt.clear_copy_range(range.start, range.size());
    }

    fn validate_region(&self, start: M::VirtAddr, size: usize) -> AxResult {
        if !self.contains_range(start, size) {
            return ax_err!(InvalidInput, "address out of range");
        }
        if !start.is_aligned_4k() || !is_aligned_4k(size) {
            return ax_err!(InvalidInput, "address not aligned");
        }
        Ok(())
    }

    /// Finds a free area that can accommodate the given size.
    ///
    /// The search starts from the given hint address, and the area should be within the given limit range.
    ///
    /// Returns the start address of the free area. Returns None if no such area is found.
    pub fn find_free_area(
        &self,
        hint: M::VirtAddr,
        size: usize,
        limit: AddrRange<M::VirtAddr>,
    ) -> Option<M::VirtAddr> {
        self.areas.find_free_area(hint, size, limit)
    }

    /// Add a new linear mapping.
    ///
    /// See [`Backend`] for more details about the mapping backends.
    ///
    /// The `flags` parameter indicates the mapping permissions and attributes.
    ///
    /// Returns an error if the address range is out of the address space or not
    /// aligned.
    pub fn map_linear(
        &mut self,
        start_vaddr: M::VirtAddr,
        start_paddr: PhysAddr,
        size: usize,
        flags: MappingFlags,
    ) -> AxResult {
        self.validate_region(start_vaddr, size)?;
        if !start_paddr.is_aligned_4k() {
            return ax_err!(InvalidInput, "address not aligned");
        }

        let offset = start_vaddr.into() - start_paddr.as_usize();
        let area = MemoryArea::new(start_vaddr, size, flags, Backend::new_linear(offset));
        self.areas
            .map(area, &mut self.pt, false)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Add a new allocation mapping.
    ///
    /// See [`Backend`] for more details about the mapping backends.
    ///
    /// The `flags` parameter indicates the mapping permissions and attributes.
    ///
    /// Returns an error if the address range is out of the address space or not
    /// aligned.
    pub fn map_alloc(
        &mut self,
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        populate: bool,
    ) -> AxResult {
        self.validate_region(start, size)?;

        let area = MemoryArea::new(start, size, flags, Backend::new_alloc(populate));
        self.areas
            .map(area, &mut self.pt, false)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// Populates the area with physical frames, returning false if the area
    /// contains unmapped area.
    pub fn populate_area(&mut self, mut start: M::VirtAddr, size: usize) -> AxResult {
        self.validate_region(start, size)?;
        let end = (start.into() + size).into();

        while let Some(area) = self.areas.find(start) {
            let backend = area.backend();
            if let Backend::Alloc { populate, .. } = backend {
                if !*populate {
                    for addr in PageIter4K::new(start, area.end().min(end)).unwrap() {
                        match self.pt.query(addr) {
                            Ok(_) => {}
                            // If the page is not mapped, try map it.
                            Err(PagingError::NotMapped) => {
                                if !backend.handle_page_fault(addr, area.flags(), &mut self.pt) {
                                    return Err(AxError::NoMemory);
                                }
                            }
                            Err(_) => return Err(AxError::BadAddress),
                        };
                    }
                }
            }
            start = area.end();
            assert!(start.is_aligned_4k());
            if start >= end {
                break;
            }
        }

        if start < end {
            // If the area is not fully mapped, we return ENOMEM.
            return ax_err!(NoMemory);
        }

        Ok(())
    }

    /// Removes mappings within the specified virtual address range.
    ///
    /// Returns an error if the address range is out of the address space or not
    /// aligned.
    pub fn unmap(&mut self, start: M::VirtAddr, size: usize) -> AxResult {
        self.validate_region(start, size)?;

        self.areas
            .unmap(start, size, &mut self.pt)
            .map_err(mapping_err_to_ax_err)?;
        Ok(())
    }

    /// To remove user area mappings from address space.
    pub fn unmap_user_areas(&mut self) -> AxResult {
        for area in self.areas.iter() {
            assert!(area.start().is_aligned_4k());
            assert!(area.size() % PAGE_SIZE_4K == 0);
            assert!(area.flags().contains(MappingFlags::USER));
            assert!(
                self.va_range
                    .contains_range(AddrRange::from_start_size(area.start(), area.size())),
                "MemorySet contains out-of-va-range area"
            );
        }
        self.areas.clear(&mut self.pt).unwrap();
        Ok(())
    }

    /// To process data in this area with the given function.
    ///
    /// Now it supports reading and writing data in the given interval.
    ///
    /// # Arguments
    /// - `start`: The start virtual address to process.
    /// - `size`: The size of the data to process.
    /// - `f`: The function to process the data, whose arguments are the start virtual address,
    ///   the offset and the size of the data.
    ///
    /// # Notes
    ///   The caller must ensure that the permission of the operation is allowed.
    fn process_area_data<F>(&self, start: M::VirtAddr, size: usize, f: F) -> AxResult
    where
        F: FnMut(M::VirtAddr, usize, usize),
    {
        Self::process_area_data_with_page_table(&self.pt, &self.va_range, start, size, f)
    }

    fn process_area_data_with_page_table<F>(
        pt: &PageTable64<M, PTE, H>,
        va_range: &AddrRange<M::VirtAddr>,
        start: M::VirtAddr,
        size: usize,
        mut f: F,
    ) -> AxResult
    where
        F: FnMut(M::VirtAddr, usize, usize),
    {
        if !va_range.contains_range(AddrRange::from_start_size(start, size)) {
            return ax_err!(InvalidInput, "address out of range");
        }
        let mut cnt = 0;
        // If start is aligned to 4K, start_align_down will be equal to start_align_up.
        let end_align_up = (start.into() + size).align_up_4k();
        for vaddr in PageIter4K::new(start.align_down_4k(), end_align_up.into())
            .expect("Failed to create page iterator")
        {
            let (mut paddr, _, _) = pt.query(vaddr).map_err(|_| AxError::BadAddress)?;

            let mut copy_size = (size - cnt).min(PAGE_SIZE_4K);

            if copy_size == 0 {
                break;
            }
            if vaddr == start.align_down_4k() && start.align_offset_4k() != 0 {
                let align_offset = start.align_offset_4k();
                copy_size = copy_size.min(PAGE_SIZE_4K - align_offset);
                paddr += align_offset;
            }
            f(H::phys_to_virt(paddr).as_usize().into(), cnt, copy_size);
            cnt += copy_size;
        }
        Ok(())
    }

    /// To read data from the address space.
    ///
    /// # Arguments
    ///
    /// * `start` - The start virtual address to read.
    /// * `buf` - The buffer to store the data.
    pub fn read(&self, start: M::VirtAddr, buf: &mut [u8]) -> AxResult {
        self.process_area_data(start, buf.len(), |src, offset, read_size| unsafe {
            core::ptr::copy_nonoverlapping(src.into() as *const u8, buf.as_mut_ptr().add(offset), read_size);
        })
    }

    /// To write data to the address space.
    ///
    /// # Arguments
    ///
    /// * `start_vaddr` - The start virtual address to write.
    /// * `buf` - The buffer to write to the address space.
    pub fn write(&self, start: M::VirtAddr, buf: &[u8]) -> AxResult {
        self.process_area_data(start, buf.len(), |dst, offset, write_size| unsafe {
            core::ptr::copy_nonoverlapping(buf.as_ptr().add(offset), dst.into() as *mut u8, write_size);
        })
    }

    /// Updates mapping within the specified virtual address range.
    ///
    /// Returns an error if the address range is out of the address space or not
    /// aligned.
    pub fn protect(&mut self, start: M::VirtAddr, size: usize, flags: MappingFlags) -> AxResult {
        // Populate the area first, which also checks the address range for us.
        self.populate_area(start, size)?;

        self.areas
            .protect(start, size, |_| Some(flags), &mut self.pt)
            .map_err(mapping_err_to_ax_err)?;

        Ok(())
    }

    /// Removes all mappings in the address space.
    pub fn clear(&mut self) {
        self.areas.clear(&mut self.pt).unwrap();
    }

    /// Checks whether an access to the specified memory region is valid.
    ///
    /// Returns `true` if the memory region given by `range` is all mapped and
    /// has proper permission flags (i.e. containing `access_flags`).
    pub fn check_region_access(
        &self,
        mut range: AddrRange<M::VirtAddr>,
        access_flags: MappingFlags,
    ) -> bool {
        for area in self.areas.iter() {
            if area.end() <= range.start {
                continue;
            }
            if area.start() > range.start {
                return false;
            }

            // This area overlaps with the memory region
            if !area.flags().contains(access_flags) {
                return false;
            }

            range.start = area.end();
            if range.is_empty() {
                return true;
            }
        }

        false
    }

    /// Handles a page fault at the given address.
    ///
    /// `access_flags` indicates the access type that caused the page fault.
    ///
    /// Returns `true` if the page fault is handled successfully (not a real
    /// fault).
    pub fn handle_page_fault(&mut self, vaddr: M::VirtAddr, access_flags: MappingFlags) -> bool {
        if !self.va_range.contains(vaddr) {
            return false;
        }
        if let Some(area) = self.areas.find(vaddr) {
            let orig_flags = area.flags();
            if !orig_flags.contains(access_flags) {
                return false;
            }
            area.backend()
                .handle_page_fault(vaddr, orig_flags, &mut self.pt)
        } else {
            false
        }
    }

    /// Clone a [`AddrSpace`] by re-mapping all [`MemoryArea`]s in a new page table and copying data in user space.
    pub fn clone_or_err(&mut self) -> AxResult<Self> {
        let mut new_aspace = Self::new_empty(self.base(), self.size())?;

        for area in self.areas.iter() {
            let backend = area.backend();
            // Remap the memory area in the new address space.
            let new_area =
                MemoryArea::new(area.start(), area.size(), area.flags(), backend.clone());
            new_aspace
                .areas
                .map(new_area, &mut new_aspace.pt, false)
                .map_err(mapping_err_to_ax_err)?;

            if matches!(backend, Backend::Linear { .. }) {
                continue;
            }
            // Copy data from old memory area to new memory area.
            for vaddr in
                PageIter4K::new(area.start(), area.end()).expect("Failed to create page iterator")
            {
                let addr = match self.pt.query(vaddr) {
                    Ok((paddr, _, _)) => paddr,
                    // If the page is not mapped, skip it.
                    Err(PagingError::NotMapped) => continue,
                    Err(_) => return Err(AxError::BadAddress),
                };
                let new_addr = match new_aspace.pt.query(vaddr) {
                    Ok((paddr, _, _)) => paddr,
                    // If the page is not mapped, try map it.
                    Err(PagingError::NotMapped) => {
                        if !backend.handle_page_fault(vaddr, area.flags(), &mut new_aspace.pt) {
                            return Err(AxError::NoMemory);
                        }
                        match new_aspace.pt.query(vaddr) {
                            Ok((paddr, _, _)) => paddr,
                            Err(_) => return Err(AxError::BadAddress),
                        }
                    }
                    Err(_) => return Err(AxError::BadAddress),
                };
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        H::phys_to_virt(addr).as_ptr(),
                        H::phys_to_virt(new_addr).as_mut_ptr(),
                        PAGE_SIZE_4K,
                    )
                };
            }
        }
        Ok(new_aspace)
    }

    /// Translates the given `VirtAddr` into `PhysAddr`.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translate(&self, vaddr: M::VirtAddr) -> Option<PhysAddr> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        self.pt
            .query(vaddr)
            .map(|(phys_addr, _, _)| {
                debug!("vaddr {:?} translate to {:?}", vaddr, phys_addr);
                phys_addr
            })
            .ok()
    }

    /// Translate&Copy the given `VirtAddr` with LENGTH len to a mutable u8 Vec through page table.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translated_byte_buffer(
        &self,
        vaddr: M::VirtAddr,
        len: usize,
    ) -> Option<Vec<&'static mut [u8]>> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        if let Some(area) = self.areas.find(vaddr) {
            if len > area.size() {
                warn!(
                    "AddrSpace translated_byte_buffer len {:#x} exceeds area length {:#x}",
                    len,
                    area.size()
                );
                return None;
            }

            let mut start: M::VirtAddr = vaddr;
            let end: M::VirtAddr = (start.into() + len).into();

            debug!(
                "start {:?} end {:?} area size {:#x}",
                start,
                end,
                area.size()
            );

            let mut v = Vec::new();
            while start < end {
                let (start_paddr, _, page_size) = self.page_table().query(start).unwrap();
                let mut end_va: M::VirtAddr = (start.align_down(page_size).into() + page_size as usize).into();
                end_va = end_va.min(end);

                v.push(unsafe {
                    core::slice::from_raw_parts_mut(
                        H::phys_to_virt(start_paddr).as_mut_ptr(),
                        (end_va.into() - start.into()).into(),
                    )
                });
                start = end_va;
            }
            Some(v)
        } else {
            None
        }
    }

    /// Translates the given `VirtAddr` into `PhysAddr`,
    /// and returns the size of the `MemoryArea` corresponding to the target vaddr.
    ///
    /// Returns `None` if the virtual address is out of range or not mapped.
    pub fn translate_and_get_limit(&self, vaddr: M::VirtAddr) -> Option<(PhysAddr, usize)> {
        if !self.va_range.contains(vaddr) {
            return None;
        }
        if let Some(area) = self.areas.find(vaddr) {
            self.pt
                .query(vaddr)
                .map(|(phys_addr, _, _)| (phys_addr, area.size()))
                .ok()
        } else {
            None
        }
    }
}

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> fmt::Debug for AddrSpace<M, PTE, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("AddrSpace")
            .field("va_range", &self.va_range)
            .field("page_table_root", &self.pt.root_paddr())
            .field("areas", &self.areas)
            .finish()
    }
}

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> Drop for AddrSpace<M, PTE, H> {
    fn drop(&mut self) {
        self.clear();
    }
}

fn mapping_err_to_ax_err(err: MappingError) -> AxError {
    warn!("Mapping error: {:?}", err);
    match err {
        MappingError::InvalidParam => AxError::InvalidInput,
        MappingError::AlreadyExists => AxError::AlreadyExists,
        MappingError::BadState => AxError::BadState,
    }
}
