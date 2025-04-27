use memory_addr::PhysAddr;
use page_table_multiarch::{PageTable64, MappingFlags, PagingHandler, PagingMetaData, GenericPTE};

use super::Backend;

impl<M: PagingMetaData, PTE: GenericPTE, H: PagingHandler> Backend<M, PTE, H> {
    /// Creates a new linear mapping backend.
    pub const fn new_linear(pa_va_offset: usize) -> Self {
        Self::Linear { pa_va_offset }
    }

    pub(crate) fn map_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        flags: MappingFlags,
        pt: &mut PageTable64<M, PTE, H>,
        pa_va_offset: usize,
    ) -> bool {
        let va_to_pa = |va: M::VirtAddr| PhysAddr::from(va.into() - pa_va_offset);
        debug!(
            "map_linear: [{:#x}, {:#x}) -> [{:#x}, {:#x}) {:?}",
            start.into(),
            start.into() + size,
            va_to_pa(start),
            va_to_pa((start.into() + size).into()),
            flags
        );
        pt.map_region(start, va_to_pa, size, flags, false, false)
            .map(|tlb| tlb.ignore()) // TLB flush on map is unnecessary, as there are no outdated mappings.
            .is_ok()
    }

    pub(crate) fn unmap_linear(
        &self,
        start: M::VirtAddr,
        size: usize,
        pt: &mut PageTable64<M, PTE, H>,
        _pa_va_offset: usize,
    ) -> bool {
        debug!("unmap_linear: [{:#x}, {:#x})", start.into(), start.into() + size);
        pt.unmap_region(start, size, true)
            .map(|tlb| tlb.ignore()) // flush each page on unmap, do not flush the entire TLB.
            .is_ok()
    }
}
