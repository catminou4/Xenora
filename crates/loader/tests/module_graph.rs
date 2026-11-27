// SPDX-License-Identifier: GPL-3.0-or-later

//! Original byte-built synthetic PE32+ fixtures for the non-executing module
//! dependency graph. No host executable or proprietary image is inspected.

mod module_graph {
    use xenora_loader::export_table::ExportTableError;
    use xenora_loader::import_table::ImportTableError;
    use xenora_loader::module_graph::{
        DataDirectory, DependencyComponent, ExportKey, FileNameError, ForwarderParseError,
        ImportOutcome, MAX_FORWARDER_HOPS, MAX_PACKAGE_FILE_NAME_LEN, MAX_PACKAGE_MODULES,
        ModuleGraph, ModuleGraphError, PackageModule, UnresolvedImportReason,
    };

    const BASE: u64 = 0x1_4000_0000;
    const OPTIONAL: usize = 0x98;
    const CODE_RVA: u32 = 0x1000;
    const DATA_RVA: u32 = 0x2000;
    const DATA_RAW: usize = 0x400;
    const DATA_LEN: usize = 0x4000;
    const DATA_LEN_U32: u32 = DATA_LEN as u32;
    const EXPORT_RVA: u32 = DATA_RVA;
    const EXPORT_LEN: u32 = 0x1000;
    const EXPORT_END: u32 = EXPORT_RVA + EXPORT_LEN;
    const EAT_RVA: u32 = EXPORT_RVA + 0x100;
    const NPT_RVA: u32 = EXPORT_RVA + 0x200;
    const EOT_RVA: u32 = EXPORT_RVA + 0x280;
    const EXPORT_STRINGS: u32 = EXPORT_RVA + 0x300;
    const IMPORT_RVA: u32 = EXPORT_END;
    const IMPORT_LEN: u32 = 0x1000;
    const IMPORT_END: u32 = IMPORT_RVA + IMPORT_LEN;
    const HEADER_EXPORT_RVA: u32 = 0x1d8;
    const DATA_SECTION_HEADER: usize = OPTIONAL + 240 + 40;

    enum ImportSpec {
        Named {
            module: String,
            symbol: String,
        },
        Ordinal {
            module: String,
            ordinal: u16,
        },
        /// A descriptor with a missing DLL-name RVA. It preserves the imports
        /// before it and then makes the ordinary import table malformed.
        BrokenDescriptor,
    }

    fn named(module: &str, symbol: &str) -> ImportSpec {
        ImportSpec::Named {
            module: module.to_owned(),
            symbol: symbol.to_owned(),
        }
    }

    fn ordinal(module: &str, ordinal: u16) -> ImportSpec {
        ImportSpec::Ordinal {
            module: module.to_owned(),
            ordinal,
        }
    }

    enum Target {
        Empty,
        Export,
        Forward(String),
    }

    struct ExportSlot {
        target: Target,
        names: Vec<String>,
    }

    struct ExportSpec {
        base: u32,
        module_name: String,
        slots: Vec<ExportSlot>,
    }

    fn name_list(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn slot(target: Target, names: &[&str]) -> ExportSlot {
        ExportSlot {
            target,
            names: name_list(names),
        }
    }

    fn export(module_name: &str, exported_names: &[&str]) -> ExportSpec {
        ExportSpec {
            base: 1,
            module_name: module_name.to_owned(),
            slots: vec![slot(Target::Export, exported_names)],
        }
    }

    fn put(pe: &mut [u8], at: usize, bytes: &[u8]) {
        pe[at..at + bytes.len()].copy_from_slice(bytes);
    }

    fn put16(pe: &mut [u8], at: usize, value: u16) {
        put(pe, at, &value.to_le_bytes());
    }

    fn put32(pe: &mut [u8], at: usize, value: u32) {
        put(pe, at, &value.to_le_bytes());
    }

    fn data_offset(rva: u32) -> usize {
        DATA_RAW + (rva - DATA_RVA) as usize
    }

    fn put32_rva(pe: &mut [u8], rva: u32, value: u32) {
        put32(pe, data_offset(rva), value);
    }

    fn allocate(cursor: &mut u32, len: usize, end: u32) -> u32 {
        let start = *cursor;
        let len = u32::try_from(len).unwrap();
        let next = start.checked_add(len).unwrap();
        *cursor = next.next_multiple_of(8);
        assert!(*cursor <= end, "fixture data does not fit in its region");
        start
    }

    fn write_string(pe: &mut [u8], cursor: &mut u32, end: u32, text: &str) -> u32 {
        let rva = allocate(cursor, text.len() + 1, end);
        put(pe, data_offset(rva), text.as_bytes());
        rva
    }

    fn write_exports(pe: &mut [u8], exports: &ExportSpec) {
        put32(pe, OPTIONAL + 112, EXPORT_RVA);
        put32(pe, OPTIONAL + 116, EXPORT_LEN);

        let mut cursor = EXPORT_STRINGS;
        let module = write_string(pe, &mut cursor, EXPORT_END, &exports.module_name);
        let mut named = Vec::new();
        for (index, export_slot) in exports.slots.iter().enumerate() {
            for name in &export_slot.names {
                named.push((name.clone(), index));
            }
        }
        named.sort();
        for (position, (name, index)) in named.iter().enumerate() {
            let name_rva = write_string(pe, &mut cursor, EXPORT_END, name);
            put32_rva(pe, NPT_RVA + 4 * position as u32, name_rva);
            put16(
                pe,
                data_offset(EOT_RVA) + 2 * position,
                u16::try_from(*index).unwrap(),
            );
        }

        for (index, export_slot) in exports.slots.iter().enumerate() {
            let value = match &export_slot.target {
                Target::Empty => 0,
                Target::Export => CODE_RVA + 4 * index as u32,
                Target::Forward(text) => write_string(pe, &mut cursor, EXPORT_END, text),
            };
            put32_rva(pe, EAT_RVA + 4 * index as u32, value);
        }

        let name_count = named.len() as u32;
        put32_rva(pe, EXPORT_RVA + 12, module);
        put32_rva(pe, EXPORT_RVA + 16, exports.base);
        put32_rva(pe, EXPORT_RVA + 20, exports.slots.len() as u32);
        put32_rva(pe, EXPORT_RVA + 24, name_count);
        put32_rva(pe, EXPORT_RVA + 28, EAT_RVA);
        put32_rva(
            pe,
            EXPORT_RVA + 32,
            if name_count == 0 { 0 } else { NPT_RVA },
        );
        put32_rva(
            pe,
            EXPORT_RVA + 36,
            if name_count == 0 { 0 } else { EOT_RVA },
        );
    }

    fn write_imports(pe: &mut [u8], imports: &[ImportSpec]) {
        put32(pe, OPTIONAL + 120, IMPORT_RVA);
        put32(pe, OPTIONAL + 124, IMPORT_LEN);
        let descriptor_len = 20 * (imports.len() as u32 + 1);
        let mut cursor = (IMPORT_RVA + descriptor_len).next_multiple_of(8);
        for (index, import) in imports.iter().enumerate() {
            let descriptor = data_offset(IMPORT_RVA) + 20 * index;
            if matches!(import, ImportSpec::BrokenDescriptor) {
                let lookup = allocate(&mut cursor, 16, IMPORT_END);
                let slot = allocate(&mut cursor, 16, IMPORT_END);
                put32(pe, descriptor, lookup);
                put32(pe, descriptor + 12, 0);
                put32(pe, descriptor + 16, slot);
                continue;
            }

            let (module, lookup_value) = match import {
                ImportSpec::Named { module, symbol } => {
                    let hint = allocate(&mut cursor, symbol.len() + 3, IMPORT_END);
                    put(pe, data_offset(hint) + 2, symbol.as_bytes());
                    (module, u64::from(hint))
                }
                ImportSpec::Ordinal { module, ordinal } => {
                    (module, (1u64 << 63) | u64::from(*ordinal))
                }
                ImportSpec::BrokenDescriptor => unreachable!(),
            };
            let module_rva = write_string(pe, &mut cursor, IMPORT_END, module);
            let lookup = allocate(&mut cursor, 16, IMPORT_END);
            let slot = allocate(&mut cursor, 16, IMPORT_END);
            put(pe, data_offset(lookup), &lookup_value.to_le_bytes());
            put(pe, data_offset(slot), &lookup_value.to_le_bytes());
            put32(pe, descriptor, lookup);
            put32(pe, descriptor + 12, module_rva);
            put32(pe, descriptor + 16, slot);
        }
    }

    fn build_image(dll: bool, imports: &[ImportSpec], exports: Option<&ExportSpec>) -> Vec<u8> {
        let mut pe = vec![0; DATA_RAW + DATA_LEN];
        put(&mut pe, 0, b"MZ");
        put32(&mut pe, 0x3c, 0x80);
        put(&mut pe, 0x80, b"PE\0\0");
        put16(&mut pe, 0x84, 0x8664);
        put16(&mut pe, 0x86, 2);
        put16(&mut pe, 0x94, 240);
        put16(&mut pe, 0x96, if dll { 0x2000 } else { 0 });
        put16(&mut pe, OPTIONAL, 0x20b);
        put32(&mut pe, OPTIONAL + 16, if dll { 0 } else { CODE_RVA });
        put(&mut pe, OPTIONAL + 24, &BASE.to_le_bytes());
        put32(&mut pe, OPTIONAL + 32, 0x1000);
        put32(&mut pe, OPTIONAL + 36, 0x200);
        put32(&mut pe, OPTIONAL + 56, DATA_RVA + DATA_LEN_U32);
        put32(&mut pe, OPTIONAL + 60, 0x200);
        put32(&mut pe, OPTIONAL + 108, 16);
        for (index, (name, rva, raw, len, flags)) in [
            (b".text", CODE_RVA, 0x200usize, 0x200usize, 0x6000_0020u32),
            (b".data", DATA_RVA, DATA_RAW, DATA_LEN, 0xc000_0040u32),
        ]
        .into_iter()
        .enumerate()
        {
            let section = OPTIONAL + 240 + 40 * index;
            put(&mut pe, section, name);
            put32(&mut pe, section + 8, len as u32);
            put32(&mut pe, section + 12, rva);
            put32(&mut pe, section + 16, len as u32);
            put32(&mut pe, section + 20, raw as u32);
            put32(&mut pe, section + 36, flags);
        }
        pe[0x200] = 0xc3;

        if !imports.is_empty() {
            write_imports(&mut pe, imports);
        }
        if let Some(exports) = exports {
            write_exports(&mut pe, exports);
        }
        pe
    }

    fn package<'a>(items: &'a [(&str, Vec<u8>)]) -> Vec<PackageModule<'a>> {
        items
            .iter()
            .map(|item| PackageModule {
                file_name: item.0,
                bytes: item.1.as_slice(),
            })
            .collect()
    }

    fn analyze(items: &[(&str, Vec<u8>)]) -> ModuleGraph {
        ModuleGraph::build(&package(items)).unwrap()
    }

    fn resolved_provider(resolution: &xenora_loader::module_graph::ImportResolution) -> usize {
        match &resolution.outcome {
            ImportOutcome::Resolved { provider, .. } => provider.index,
            outcome => panic!("import is not resolved: {outcome:?}"),
        }
    }

    fn unresolved_reason(
        resolution: &xenora_loader::module_graph::ImportResolution,
    ) -> &UnresolvedImportReason {
        match &resolution.outcome {
            ImportOutcome::Unresolved(reason) => reason,
            outcome => panic!("import unexpectedly resolved: {outcome:?}"),
        }
    }

    fn edge_tuples(graph: &ModuleGraph) -> Vec<(usize, usize)> {
        graph
            .dependency_edges()
            .iter()
            .map(|edge| (edge.consumer, edge.provider))
            .collect()
    }

    #[test]
    fn direct_transitive_and_shared_dependencies_are_deterministic() {
        let shared = build_image(true, &[], Some(&export("shared.dll", &["Value"])));
        let middle = build_image(
            true,
            &[named("shared.dll", "Value")],
            Some(&export("middle.dll", &["Middle"])),
        );
        let other = build_image(
            true,
            &[named("shared.dll", "Value")],
            Some(&export("other.dll", &["Other"])),
        );
        let root = build_image(
            false,
            &[named("middle.dll", "Middle"), named("other.dll", "Other")],
            None,
        );
        let items = [
            ("root.exe", root),
            ("middle.dll", middle),
            ("shared.dll", shared),
            ("other.dll", other),
        ];
        let graph = analyze(&items);

        assert!(graph.inspection_is_complete());
        assert_eq!(graph.import_resolutions().len(), 4);
        assert!(
            graph
                .import_resolutions()
                .iter()
                .all(|resolution| matches!(resolution.outcome, ImportOutcome::Resolved { .. }))
        );
        assert_eq!(resolved_provider(&graph.import_resolutions()[0]), 1);
        assert_eq!(resolved_provider(&graph.import_resolutions()[1]), 3);
        assert_eq!(edge_tuples(&graph), vec![(0, 1), (0, 3), (1, 2), (3, 2)]);
        assert_eq!(
            graph.dependency_order(),
            [
                DependencyComponent::Acyclic(2),
                DependencyComponent::Acyclic(1),
                DependencyComponent::Acyclic(3),
                DependencyComponent::Acyclic(0),
            ]
        );
    }

    #[test]
    fn ordinals_use_the_export_base_and_empty_slots_are_absent() {
        let spec = ExportSpec {
            base: 11,
            module_name: "ordinal.dll".to_owned(),
            slots: vec![
                slot(Target::Export, &[]),
                slot(Target::Empty, &["Gap"]),
                slot(Target::Export, &[]),
            ],
        };
        let library = build_image(true, &[], Some(&spec));
        let root = build_image(
            false,
            &[
                ordinal("ordinal.dll", 11),
                ordinal("ordinal.dll", 12),
                ordinal("ordinal.dll", 14),
                named("ordinal.dll", "Gap"),
            ],
            None,
        );
        let items = [("root.exe", root), ("ordinal.dll", library)];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        match &resolutions[0].outcome {
            ImportOutcome::Resolved {
                provider,
                export,
                ordinal,
                ..
            } => {
                assert_eq!(provider.index, 1);
                assert_eq!(*export, ExportKey::Ordinal(11));
                assert_eq!(*ordinal, 11);
            }
            outcome => panic!("ordinal 11 did not resolve: {outcome:?}"),
        }
        for (index, export) in [
            (1, ExportKey::Ordinal(12)),
            (3, ExportKey::Name("Gap".to_owned())),
        ] {
            match unresolved_reason(&resolutions[index]) {
                UnresolvedImportReason::EmptyExportSlot {
                    provider,
                    export: actual,
                    ordinal,
                } => {
                    assert_eq!(provider.index, 1);
                    assert_eq!(*actual, export);
                    assert_eq!(*ordinal, 12);
                }
                reason => panic!("empty slot was not diagnosed for import {index}: {reason:?}"),
            }
        }
        match unresolved_reason(&resolutions[2]) {
            UnresolvedImportReason::MissingExport { provider, export } => {
                assert_eq!(provider.index, 1);
                assert_eq!(*export, ExportKey::Ordinal(14));
            }
            reason => panic!("missing ordinal was not diagnosed: {reason:?}"),
        }
    }

    #[test]
    fn module_names_fold_ascii_case_but_export_names_and_embedded_dll_names_do_not() {
        let provider = build_image(
            true,
            &[],
            Some(&export("embedded-other.dll", &["CaseSensitive"])),
        );
        let root = build_image(
            false,
            &[
                named("mixed.DLL", "CaseSensitive"),
                named("MIXED.dll", "casesensitive"),
                named("embedded-other.dll", "CaseSensitive"),
            ],
            None,
        );
        let items = [("root.exe", root), ("MiXeD.dll", provider)];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        assert_eq!(graph.modules()[1].canonical_file_name, "mixed.dll");
        assert_eq!(
            graph.modules()[1].exports.as_ref().unwrap().module,
            "embedded-other.dll"
        );
        assert_eq!(resolutions[0].normalized_module, "mixed.dll");
        assert_eq!(resolved_provider(&resolutions[0]), 1);
        // The same provider edge is retained even though the second import's
        // exact-case symbol is absent; the nonexistent embedded name has no edge.
        assert_eq!(edge_tuples(&graph), vec![(0, 1)]);
        match unresolved_reason(&resolutions[1]) {
            UnresolvedImportReason::MissingExport { provider, export } => {
                assert_eq!(provider.index, 1);
                assert_eq!(*export, ExportKey::Name("casesensitive".to_owned()));
            }
            reason => panic!("export case mismatch was not diagnosed: {reason:?}"),
        }
        match unresolved_reason(&resolutions[2]) {
            UnresolvedImportReason::MissingModule { module } => {
                assert_eq!(module, "embedded-other.dll")
            }
            reason => panic!("embedded export-table name was used as an identity: {reason:?}"),
        }
    }

    #[test]
    fn missing_modules_and_no_extension_reference_normalization_are_deterministic() {
        let bare = build_image(true, &[], Some(&export("bare", &["Needed"])));
        let root = build_image(
            false,
            &[
                named("absent", "X"),
                named("bare.", "Needed"),
                named("bare", "Needed"),
            ],
            None,
        );
        let items = [("root.exe", root), ("bare", bare)];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        assert_eq!(resolutions[0].normalized_module, "absent.dll");
        assert!(matches!(
            unresolved_reason(&resolutions[0]),
            UnresolvedImportReason::MissingModule { module } if module == "absent.dll"
        ));
        assert_eq!(resolutions[1].normalized_module, "bare");
        assert_eq!(resolved_provider(&resolutions[1]), 1);
        assert_eq!(resolutions[2].normalized_module, "bare.dll");
        assert!(matches!(
            unresolved_reason(&resolutions[2]),
            UnresolvedImportReason::MissingModule { module } if module == "bare.dll"
        ));
    }

    #[test]
    fn circular_components_are_ordered_against_external_dependencies() {
        let core = build_image(true, &[], Some(&export("core.dll", &["Core"])));
        let a = build_image(
            true,
            &[named("core.dll", "Core"), named("b.dll", "B")],
            Some(&export("a.dll", &["A"])),
        );
        let b = build_image(true, &[named("a.dll", "A")], Some(&export("b.dll", &["B"])));
        let root = build_image(false, &[named("a.dll", "A")], None);
        let items = [
            ("root.exe", root),
            ("a.dll", a),
            ("b.dll", b),
            ("core.dll", core),
        ];
        let graph = analyze(&items);

        assert_eq!(edge_tuples(&graph), vec![(0, 1), (1, 2), (1, 3), (2, 1)]);
        assert_eq!(
            graph.dependency_order(),
            [
                DependencyComponent::Acyclic(3),
                DependencyComponent::Circular(vec![1, 2]),
                DependencyComponent::Acyclic(0),
            ]
        );
    }

    #[test]
    fn circular_components_include_self_dependencies_and_have_no_fake_order() {
        let a = build_image(true, &[named("b.dll", "B")], Some(&export("a.dll", &["A"])));
        let b = build_image(true, &[named("a.dll", "A")], Some(&export("b.dll", &["B"])));
        let self_dependent = build_image(
            true,
            &[named("self.dll", "Own")],
            Some(&export("self.dll", &["Own"])),
        );
        let solo = build_image(true, &[], Some(&export("solo.dll", &["Solo"])));
        let items = [
            ("b.dll", b),
            ("a.dll", a),
            ("self.dll", self_dependent),
            ("solo.dll", solo),
        ];
        let graph = analyze(&items);

        assert!(graph.inspection_is_complete());
        assert_eq!(edge_tuples(&graph), vec![(0, 1), (1, 0), (2, 2)]);
        assert_eq!(
            graph.dependency_order(),
            [
                DependencyComponent::Circular(vec![0, 1]),
                DependencyComponent::Circular(vec![2]),
                DependencyComponent::Acyclic(3),
            ]
        );
    }

    #[test]
    fn malformed_import_tables_keep_prior_imports_and_structured_errors() {
        let library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        let root = build_image(
            false,
            &[named("lib.dll", "X"), ImportSpec::BrokenDescriptor],
            None,
        );
        let items = [("root.exe", root), ("lib.dll", library)];
        let graph = analyze(&items);
        let root_node = &graph.modules()[0];

        assert_eq!(root_node.imports.len(), 1);
        assert_eq!(
            root_node.import_error,
            Some(ImportTableError::IncompleteImportDescriptor {
                rva: IMPORT_RVA + 20,
            })
        );
        assert!(!graph.inspection_is_complete());
        assert_eq!(resolved_provider(&graph.import_resolutions()[0]), 1);
        assert_eq!(edge_tuples(&graph), vec![(0, 1)]);
    }

    #[test]
    fn malformed_export_tables_are_preserved_but_not_used_for_resolution() {
        const UNBACKED_NAME_RVA: u32 = 0x7000;
        let mut library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        put32_rva(&mut library, NPT_RVA, UNBACKED_NAME_RVA);
        let root = build_image(false, &[named("lib.dll", "X")], None);
        let items = [("root.exe", root), ("lib.dll", library)];
        let graph = analyze(&items);
        let library_node = &graph.modules()[1];

        assert_eq!(
            library_node.export_error,
            Some(ExportTableError::MalformedExportName {
                rva: UNBACKED_NAME_RVA,
            })
        );
        assert_eq!(library_node.exports.as_ref().unwrap().entries.len(), 1);
        assert!(!graph.inspection_is_complete());
        assert_eq!(edge_tuples(&graph), vec![(0, 1)]);
        assert!(matches!(
            unresolved_reason(&graph.import_resolutions()[0]),
            UnresolvedImportReason::ProviderExportsMalformed { provider } if provider.index == 1
        ));
    }

    #[test]
    fn virtual_zero_fill_is_not_backing_for_import_tables() {
        const UNBACKED_IMPORT_RVA: u32 = DATA_RVA + DATA_LEN_U32;
        let library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        let mut root = build_image(false, &[named("lib.dll", "X")], None);
        // Expand .data's virtual span beyond its raw bytes. The added bytes are
        // zero-filled memory in a real mapping, not bytes backed by the image.
        put32(&mut root, DATA_SECTION_HEADER + 8, DATA_LEN_U32 + 0x1000);
        put32(&mut root, OPTIONAL + 56, DATA_RVA + DATA_LEN_U32 + 0x1000);
        put32(&mut root, OPTIONAL + 120, UNBACKED_IMPORT_RVA);
        let items = [("root.exe", root), ("lib.dll", library)];
        let graph = analyze(&items);

        assert_eq!(
            graph.modules()[0].import_error,
            Some(ImportTableError::OutOfBounds {
                rva: UNBACKED_IMPORT_RVA,
            })
        );
        assert!(graph.modules()[0].imports.is_empty());
        assert!(!graph.inspection_is_complete());
    }

    #[test]
    fn data_directories_honor_number_of_rva_and_sizes() {
        let mut library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        put32(&mut library, OPTIONAL + 108, 0);
        let mut root = build_image(false, &[named("lib.dll", "X")], None);
        put32(&mut root, OPTIONAL + 108, 1);
        let items = [("root.exe", root), ("lib.dll", library)];
        let graph = analyze(&items);

        assert!(graph.inspection_is_complete());
        assert!(graph.modules()[0].imports.is_empty());
        assert!(graph.modules()[0].import_error.is_none());
        assert!(graph.modules()[1].exports.is_none());
        assert!(graph.modules()[1].export_error.is_none());
        assert!(graph.import_resolutions().is_empty());
    }

    #[test]
    fn data_directory_entries_are_checked_against_the_declared_optional_header() {
        let mut root = build_image(false, &[named("lib.dll", "X")], None);
        // Keep only the fixed PE32+ optional header plus the export directory.
        // Move the section table accordingly; the import directory entry would
        // occupy its first bytes and is therefore truncated by the declared header.
        put16(&mut root, 0x94, 120);
        let old_sections = OPTIONAL + 240;
        let new_sections = OPTIONAL + 120;
        root.copy_within(old_sections..old_sections + 80, new_sections);
        let library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        let items = [("root.exe", root), ("lib.dll", library)];

        match ModuleGraph::build(&package(&items)) {
            Err(ModuleGraphError::TruncatedDataDirectory {
                module,
                directory: DataDirectory::Import,
            }) => assert_eq!(module.index, 0),
            result => panic!("truncated import directory was not diagnosed: {result:?}"),
        }
    }

    #[test]
    fn image_headers_are_valid_backing_for_an_export_directory() {
        let mut library = build_image(true, &[], Some(&export("lib.dll", &["X"])));
        let source = data_offset(EXPORT_RVA);
        library.copy_within(source..source + 40, HEADER_EXPORT_RVA as usize);
        put32(&mut library, OPTIONAL + 112, HEADER_EXPORT_RVA);
        put32(&mut library, OPTIONAL + 116, EXPORT_END - HEADER_EXPORT_RVA);
        // Once the declared export range starts in the headers it also covers
        // the fixture's .text RVA. Move the inventory target outside that
        // range so the parser does not classify it as a forwarder string.
        put32_rva(&mut library, EAT_RVA, DATA_RVA + DATA_LEN_U32 - 4);
        let root = build_image(false, &[named("lib.dll", "X")], None);
        let items = [("root.exe", root), ("lib.dll", library)];
        let graph = analyze(&items);

        assert!(graph.inspection_is_complete());
        assert_eq!(resolved_provider(&graph.import_resolutions()[0]), 1);
    }

    #[test]
    fn forwarders_resolve_named_and_ordinal_targets_within_the_package() {
        let final_spec = ExportSpec {
            base: 7,
            module_name: "final.dll".to_owned(),
            slots: vec![slot(Target::Export, &["Impl"])],
        };
        let final_library = build_image(true, &[], Some(&final_spec));
        let middle_spec = ExportSpec {
            base: 1,
            module_name: "middle.dll".to_owned(),
            slots: vec![
                slot(Target::Forward("final.Impl".to_owned()), &["Forwarded"]),
                slot(
                    Target::Forward("final.#7".to_owned()),
                    &["ForwardedOrdinal"],
                ),
            ],
        };
        let middle = build_image(true, &[], Some(&middle_spec));
        let root = build_image(
            false,
            &[
                named("middle.dll", "Forwarded"),
                named("middle.dll", "ForwardedOrdinal"),
            ],
            None,
        );
        let items = [
            ("root.exe", root),
            ("middle.dll", middle),
            ("final.dll", final_library),
        ];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        assert_eq!(edge_tuples(&graph), vec![(0, 1), (1, 2)]);
        assert_eq!(
            graph.dependency_order(),
            [
                DependencyComponent::Acyclic(2),
                DependencyComponent::Acyclic(1),
                DependencyComponent::Acyclic(0),
            ]
        );
        for (index, key) in [
            (0, ExportKey::Name("Impl".to_owned())),
            (1, ExportKey::Ordinal(7)),
        ] {
            let resolution = &resolutions[index];
            assert_eq!(resolution.forwarder_chain.len(), 1);
            assert_eq!(resolution.forwarder_chain[0].target_module, "final.dll");
            match &resolution.outcome {
                ImportOutcome::Resolved {
                    provider,
                    export,
                    ordinal,
                    rva,
                } => {
                    assert_eq!(provider.index, 2);
                    assert_eq!(*export, key);
                    assert_eq!(*ordinal, 7);
                    assert_eq!(*rva, CODE_RVA);
                }
                outcome => panic!("forwarder {index} did not resolve: {outcome:?}"),
            }
        }
    }

    #[test]
    fn forwarder_missing_module_identifies_the_forwarding_step() {
        let middle_spec = ExportSpec {
            base: 1,
            module_name: "middle.dll".to_owned(),
            slots: vec![slot(
                Target::Forward("absent.Impl".to_owned()),
                &["Forwarded"],
            )],
        };
        let middle = build_image(true, &[], Some(&middle_spec));
        let root = build_image(false, &[named("middle.dll", "Forwarded")], None);
        let items = [("root.exe", root), ("middle.dll", middle)];
        let graph = analyze(&items);
        let resolution = &graph.import_resolutions()[0];

        assert!(resolution.forwarder_chain.is_empty());
        match unresolved_reason(resolution) {
            UnresolvedImportReason::ForwarderMissingModule { step } => {
                assert_eq!(step.provider.index, 1);
                assert_eq!(step.export, ExportKey::Name("Forwarded".to_owned()));
                assert_eq!(step.target, "absent.Impl");
                assert_eq!(step.target_module, "absent.dll");
                assert_eq!(step.target_export, ExportKey::Name("Impl".to_owned()));
            }
            reason => panic!("missing forwarder target was not diagnosed: {reason:?}"),
        }
    }

    #[test]
    fn forwarder_cycles_are_detected_before_unbounded_resolution() {
        let ca_spec = ExportSpec {
            base: 1,
            module_name: "ca.dll".to_owned(),
            slots: vec![slot(Target::Forward("cb.X".to_owned()), &["A"])],
        };
        let cb_spec = ExportSpec {
            base: 1,
            module_name: "cb.dll".to_owned(),
            slots: vec![slot(Target::Forward("ca.A".to_owned()), &["X"])],
        };
        let ca = build_image(true, &[], Some(&ca_spec));
        let cb = build_image(true, &[], Some(&cb_spec));
        let root = build_image(false, &[named("ca.dll", "A")], None);
        let items = [("root.exe", root), ("ca.dll", ca), ("cb.dll", cb)];
        let graph = analyze(&items);
        let resolution = &graph.import_resolutions()[0];

        assert_eq!(resolution.forwarder_chain.len(), 1);
        match unresolved_reason(resolution) {
            UnresolvedImportReason::ForwarderCycle { step } => {
                assert_eq!(step.provider.index, 2);
                assert_eq!(step.target_module, "ca.dll");
                assert_eq!(step.target_export, ExportKey::Name("A".to_owned()));
            }
            reason => panic!("forwarder cycle was not diagnosed: {reason:?}"),
        }
        assert!(graph.dependency_order().iter().any(
            |item| matches!(item, DependencyComponent::Circular(members) if members == &[1, 2])
        ));
    }

    #[test]
    fn malformed_forwarders_have_precise_structured_errors() {
        let spec = ExportSpec {
            base: 1,
            module_name: "badfw.dll".to_owned(),
            slots: vec![
                slot(Target::Forward("NoSeparator".to_owned()), &["NoSeparator"]),
                slot(Target::Forward(".Name".to_owned()), &["EmptyModule"]),
                slot(Target::Forward("dll.".to_owned()), &["EmptySymbol"]),
                slot(Target::Forward("dll.#".to_owned()), &["EmptyOrdinal"]),
                slot(Target::Forward("dll.#x".to_owned()), &["BadOrdinal"]),
                slot(
                    Target::Forward("dll.#4294967296".to_owned()),
                    &["OverflowOrdinal"],
                ),
            ],
        };
        let provider = build_image(true, &[], Some(&spec));
        let root = build_image(
            false,
            &[
                named("badfw.dll", "NoSeparator"),
                named("badfw.dll", "EmptyModule"),
                named("badfw.dll", "EmptySymbol"),
                named("badfw.dll", "EmptyOrdinal"),
                named("badfw.dll", "BadOrdinal"),
                named("badfw.dll", "OverflowOrdinal"),
            ],
            None,
        );
        let items = [("root.exe", root), ("badfw.dll", provider)];
        let graph = analyze(&items);
        let errors: Vec<_> = graph
            .import_resolutions()
            .iter()
            .map(|resolution| match unresolved_reason(resolution) {
                UnresolvedImportReason::MalformedForwarder { error, .. } => *error,
                reason => panic!("forwarder was not malformed: {reason:?}"),
            })
            .collect();

        assert_eq!(
            errors,
            [
                ForwarderParseError::MissingSeparator,
                ForwarderParseError::EmptyModuleName,
                ForwarderParseError::EmptySymbolName,
                ForwarderParseError::EmptyOrdinal,
                ForwarderParseError::InvalidOrdinalDigit,
                ForwarderParseError::OrdinalOverflow,
            ]
        );
    }

    #[test]
    fn forwarder_resolution_stops_at_the_documented_hop_bound() {
        let mut items: Vec<(String, Vec<u8>)> = Vec::new();
        let root = build_image(false, &[named("f0.dll", "X")], None);
        items.push(("root.exe".to_owned(), root));
        for index in 0..=MAX_FORWARDER_HOPS {
            let module_name = format!("f{index}.dll");
            let target = format!("f{}.X", index + 1);
            let spec = ExportSpec {
                base: 1,
                module_name: module_name.clone(),
                slots: vec![slot(Target::Forward(target), &["X"])],
            };
            let image = build_image(true, &[], Some(&spec));
            items.push((module_name, image));
        }
        let final_name = format!("f{}.dll", MAX_FORWARDER_HOPS + 1);
        let final_image = build_image(true, &[], Some(&export(&final_name, &["X"])));
        items.push((final_name, final_image));
        let package: Vec<_> = items
            .iter()
            .map(|item| PackageModule {
                file_name: item.0.as_str(),
                bytes: item.1.as_slice(),
            })
            .collect();
        let graph = ModuleGraph::build(&package).unwrap();
        let resolution = &graph.import_resolutions()[0];

        assert_eq!(resolution.forwarder_chain.len(), MAX_FORWARDER_HOPS);
        match unresolved_reason(resolution) {
            UnresolvedImportReason::ForwarderHopLimit { step } => {
                assert_eq!(step.provider.file_name, "f8.dll");
                assert_eq!(step.target_module, "f9.dll");
            }
            reason => panic!("forwarder hop limit was not diagnosed: {reason:?}"),
        }
    }

    #[test]
    fn forwarder_resolution_accepts_a_terminal_export_at_the_hop_bound() {
        let mut items: Vec<(String, Vec<u8>)> = Vec::new();
        let root = build_image(false, &[named("f0.dll", "X")], None);
        items.push(("root.exe".to_owned(), root));
        for index in 0..MAX_FORWARDER_HOPS {
            let module_name = format!("f{index}.dll");
            let target = format!("f{}.X", index + 1);
            let spec = ExportSpec {
                base: 1,
                module_name: module_name.clone(),
                slots: vec![slot(Target::Forward(target), &["X"])],
            };
            items.push((module_name, build_image(true, &[], Some(&spec))));
        }
        let final_name = format!("f{MAX_FORWARDER_HOPS}.dll");
        let final_image = build_image(true, &[], Some(&export(&final_name, &["X"])));
        items.push((final_name, final_image));
        let package: Vec<_> = items
            .iter()
            .map(|item| PackageModule {
                file_name: item.0.as_str(),
                bytes: item.1.as_slice(),
            })
            .collect();
        let graph = ModuleGraph::build(&package).unwrap();
        let resolution = &graph.import_resolutions()[0];

        assert_eq!(resolution.forwarder_chain.len(), MAX_FORWARDER_HOPS);
        assert_eq!(resolved_provider(resolution), MAX_FORWARDER_HOPS + 1);
    }

    #[test]
    fn path_like_and_api_set_references_are_not_package_files() {
        let api_file = "api-ms-win-core-test-l1-1-0.dll";
        let api_module = build_image(true, &[], Some(&export(api_file, &["X"])));
        let ordinary = build_image(true, &[], Some(&export("a.dll", &["X"])));
        let root = build_image(
            false,
            &[
                named(api_file, "X"),
                named("API-MS-WIN-CORE-TEST-L1-1-0", "X"),
                named("plugins\\a.dll", "X"),
                named("a.dll", "X"),
            ],
            None,
        );
        let items = [
            ("root.exe", root),
            (api_file, api_module),
            ("a.dll", ordinary),
        ];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        assert_eq!(edge_tuples(&graph), vec![(0, 2)]);
        match unresolved_reason(&resolutions[0]) {
            UnresolvedImportReason::ApiSetContract { contract } => {
                assert_eq!(contract, api_file);
            }
            reason => panic!("API-set file was treated as a package module: {reason:?}"),
        }
        match unresolved_reason(&resolutions[1]) {
            UnresolvedImportReason::ApiSetContract { contract } => {
                assert_eq!(contract, api_file);
            }
            reason => panic!("mixed-case API-set contract was not recognized: {reason:?}"),
        }
        match unresolved_reason(&resolutions[2]) {
            UnresolvedImportReason::UnsupportedModulePath { module } => {
                assert_eq!(module, "plugins\\a.dll");
            }
            reason => panic!("path-like module reference was not diagnosed: {reason:?}"),
        }
        assert_eq!(resolved_provider(&resolutions[3]), 2);
    }

    #[test]
    fn forwarders_use_the_last_dot_and_unused_forwarders_add_no_edge() {
        let final_library = build_image(true, &[], Some(&export("my.x64.dll", &["Func"])));
        let unused_target = build_image(true, &[], Some(&export("unused.dll", &["Target"])));
        let middle_spec = ExportSpec {
            base: 1,
            module_name: "middle.dll".to_owned(),
            slots: vec![
                slot(Target::Export, &["Direct"]),
                slot(
                    Target::Forward("my.x64.dll.Func".to_owned()),
                    &["Forwarded"],
                ),
                slot(Target::Forward("unused.Target".to_owned()), &["Unused"]),
            ],
        };
        let middle = build_image(true, &[], Some(&middle_spec));
        let root = build_image(
            false,
            &[
                named("middle.dll", "Direct"),
                named("middle.dll", "Forwarded"),
            ],
            None,
        );
        let items = [
            ("root.exe", root),
            ("middle.dll", middle),
            ("my.x64.dll", final_library),
            ("unused.dll", unused_target),
        ];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        assert_eq!(edge_tuples(&graph), vec![(0, 1), (1, 2)]);
        assert_eq!(resolved_provider(&resolutions[0]), 1);
        assert_eq!(resolutions[1].forwarder_chain.len(), 1);
        assert_eq!(
            resolutions[1].forwarder_chain[0].target_module,
            "my.x64.dll"
        );
        assert_eq!(
            resolutions[1].forwarder_chain[0].target_export,
            ExportKey::Name("Func".to_owned())
        );
        assert_eq!(resolved_provider(&resolutions[1]), 2);
        assert!(!edge_tuples(&graph).contains(&(1, 3)));
    }

    #[test]
    fn forwarder_api_sets_and_paths_have_explicit_diagnostics() {
        let api_file = "api-ms-win-core-forward-l1-1-0.dll";
        let api_module = build_image(true, &[], Some(&export(api_file, &["X"])));
        let middle_spec = ExportSpec {
            base: 1,
            module_name: "middle.dll".to_owned(),
            slots: vec![
                slot(
                    Target::Forward("api-ms-win-core-forward-l1-1-0.X".to_owned()),
                    &["Api"],
                ),
                slot(Target::Forward("plugins\\target.X".to_owned()), &["Path"]),
            ],
        };
        let middle = build_image(true, &[], Some(&middle_spec));
        let root = build_image(
            false,
            &[named("middle.dll", "Api"), named("middle.dll", "Path")],
            None,
        );
        let items = [
            ("root.exe", root),
            ("middle.dll", middle),
            (api_file, api_module),
        ];
        let graph = analyze(&items);
        let resolutions = graph.import_resolutions();

        match unresolved_reason(&resolutions[0]) {
            UnresolvedImportReason::ForwarderApiSetContract { step } => {
                assert_eq!(step.provider.index, 1);
                assert_eq!(step.target_module, api_file);
                assert_eq!(step.target_export, ExportKey::Name("X".to_owned()));
            }
            reason => panic!("forwarded API-set contract was not diagnosed: {reason:?}"),
        }
        match unresolved_reason(&resolutions[1]) {
            UnresolvedImportReason::MalformedForwarder { error, .. } => {
                assert_eq!(*error, ForwarderParseError::ModulePathSeparator);
            }
            reason => panic!("forwarded module path was not diagnosed: {reason:?}"),
        }
        assert_eq!(edge_tuples(&graph), vec![(0, 1)]);
    }

    #[test]
    fn input_file_names_are_conservatively_validated() {
        let image = build_image(true, &[], None);
        let cases = [
            (String::new(), FileNameError::Empty),
            (
                "a".repeat(MAX_PACKAGE_FILE_NAME_LEN + 1),
                FileNameError::TooLong {
                    len: MAX_PACKAGE_FILE_NAME_LEN + 1,
                    max: MAX_PACKAGE_FILE_NAME_LEN,
                },
            ),
            ("bad/name.dll".to_owned(), FileNameError::PathSeparator),
            ("bad\\name.dll".to_owned(), FileNameError::PathSeparator),
            ("bad:name.dll".to_owned(), FileNameError::DriveSeparator),
            ("bad\tname.dll".to_owned(), FileNameError::NotPrintableAscii),
            ("é.dll".to_owned(), FileNameError::NotPrintableAscii),
            ("bare.".to_owned(), FileNameError::TrailingDot),
        ];
        for (file_name, reason) in cases {
            let package = [PackageModule {
                file_name: &file_name,
                bytes: image.as_slice(),
            }];
            match ModuleGraph::build(&package) {
                Err(ModuleGraphError::InvalidFileName {
                    index: 0,
                    reason: actual,
                    ..
                }) => assert_eq!(actual, reason),
                result => panic!("invalid name {file_name:?} was not rejected: {result:?}"),
            }
        }
    }

    #[test]
    fn duplicate_canonical_identities_and_module_count_are_rejected() {
        let image = build_image(true, &[], None);
        let duplicate_items = [("same.dll", image.clone()), ("SAME.DLL", image.clone())];
        let duplicate_package = package(&duplicate_items);
        assert_eq!(
            ModuleGraph::build(&duplicate_package),
            Err(ModuleGraphError::DuplicateModuleIdentity {
                first_index: 0,
                second_index: 1,
                identity: "same.dll".to_owned(),
            })
        );

        let count = MAX_PACKAGE_MODULES + 1;
        let names: Vec<_> = (0..count).map(|index| format!("m{index}.dll")).collect();
        let images = vec![image; count];
        let package: Vec<_> = names
            .iter()
            .zip(&images)
            .map(|(name, image)| PackageModule {
                file_name: name,
                bytes: image.as_slice(),
            })
            .collect();
        assert_eq!(
            ModuleGraph::build(&package),
            Err(ModuleGraphError::TooManyModules {
                count,
                max: MAX_PACKAGE_MODULES,
            })
        );
    }
}
