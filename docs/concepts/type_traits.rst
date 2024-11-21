.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _type_traits:

********************************************************************
 Implement traits for custom types
********************************************************************

Overview
========

This interface is designed to enable users to provide additional type trait information to rocPRIM, facilitating better compatibility with custom types.

Custom types that implement arithmetic operators can behave like built-in arithmetic types; however, algorithms in rocPRIM may still interpret them as generic `struct` or `class` types. For some algorithms, processing methods differ between floating-point and integral types. Accurately describing custom types, therefore, is essential not only for performance optimization but also for ensuring correctness. The rocPRIM type traits interface allows users to supply custom trait information for their types, greatly enhancing compatibility between these types and rocPRIM algorithms.

This interface is strict but flexible, similar to operator overloading. Users should only implement traits as required by specific algorithms, and some traits cannot be defined if they can be inferred from others. Adhering to these rules is crucial for successful compilation. Fortunately, the interface provides clear error messages to guide users when issues arise.


Interface
=========

.. doxygengroup::  type_traits_interfaces
  :content-only:
  :members:


Available traits
================

.. doxygengroup::  available_traits
  :content-only:
  :members:

Type traits wrappers
====================
.. doxygengroup::  rocprim_type_traits_wrapper
  :content-only:
  :no-link:

Types with predefined traits
============================

.. doxygengroup::  rocprim_pre_defined_traits
  :content-only:
  :members:
  :outline:
  :no-link:
  
  