# Address space generics
Address space is an important and common concept for various kernels and hypervisors.
Almost everything must run in a specific address space.
Address space (Aspace) includs three aspects:
1. Total range of this aspace.
2. A series of areas in the total range.
3. Pagetable to map this aspace to another.
