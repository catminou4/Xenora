// SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt::Write;

use xenora_decoder::{
    DecodedInstruction, ExportTarget, Instruction, InterpolationPhase, VectorAluOp, VectorSource,
};

use super::{
    Architecture, Binary, CompileError, CompiledShader, Value, VectorBinary, decode_fixture,
};

/// Entry point of the translated vertex stage.
pub const VERTEX_ENTRY_POINT: &str = "xenora_vertex";
/// Entry point of the translated pixel stage.
pub const PIXEL_ENTRY_POINT: &str = "xenora_pixel";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Vertex,
    Pixel,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Pixel => "pixel",
        }
    }
}

#[derive(Clone, Copy)]
enum Float4Input {
    Position,
    Color,
}

#[derive(Clone, Copy)]
enum Op {
    VertexInput(Float4Input, u8),
    Constant(u32),
    Interpolate(u8),
    Binary(Binary, Value, Value),
    SampleLodZero {
        u: Value,
        v: Value,
        component: u8,
        texture_index: u8,
        sampler_index: u8,
    },
}

struct Export {
    offset: usize,
    values: [Value; 4],
}

#[derive(Default)]
struct StageIr {
    values: Vec<(Option<usize>, Op)>,
    position: Option<Export>,
    parameter: Option<Export>,
    color: Option<Export>,
}

impl StageIr {
    fn push(&mut self, offset: Option<usize>, op: Op) -> Value {
        let value = Value(self.values.len());
        self.values.push((offset, op));
        value
    }
}

/// Compiles a linked synthetic gfx1030 pipeline with one RGBA8 color attachment.
///
/// Vertex VGPRs 0–3 receive one clip-space position from buffer 0 and VGPRs 4–7
/// receive one color from buffer 1. Each buffer contains one float4 per
/// vertex. Pixel VGPRs 0 and 1 are reserved for the I and J barycentric
/// coordinates. Adjacent interpolation pairs for PARAM0 lower to Metal's
/// center-perspective interpolation under this ABI. Vertex FP32 addition and
/// multiplication use separate operations and require the backend's safe and
/// precise Metal math modes.
pub fn compile_graphics(
    architecture: Architecture,
    vertex: &[u8],
    pixel: &[u8],
) -> Result<CompiledShader, CompileError> {
    let compile_stage = |stage: Stage, bytes: &[u8]| {
        let result = decode_fixture(architecture, bytes).and_then(|decoded| {
            let lifted = lift(stage, &decoded, bytes.len())?;
            Ok((decoded, lifted))
        });
        result.map_err(|error| CompileError::GraphicsStage {
            stage: stage.name(),
            error: Box::new(error),
        })
    };
    let (vertex_decoded, vertex) = compile_stage(Stage::Vertex, vertex)?;
    let (pixel_decoded, pixel) = compile_stage(Stage::Pixel, pixel)?;
    let position = vertex.position.as_ref().ok_or(CompileError::GraphicsLink {
        reason: "vertex stage requires a POS0 export",
    })?;
    let color = pixel.color.as_ref().ok_or(CompileError::GraphicsLink {
        reason: "pixel stage requires an MRT0 export",
    })?;
    let parameter = vertex.parameter.as_ref();
    let interpolates = pixel
        .values
        .iter()
        .any(|(_, op)| matches!(op, Op::Interpolate(_)));
    if parameter.is_some() != interpolates {
        return Err(CompileError::GraphicsLink {
            reason: "vertex PARAM0 export and pixel attr0 interpolation must both be present or absent",
        });
    }

    let has_sampling = pixel
        .values
        .iter()
        .any(|(_, op)| matches!(op, Op::SampleLodZero { .. }));

    let mut disassembly = String::new();
    for (name, instructions) in [("vertex", vertex_decoded), ("pixel", pixel_decoded)] {
        let _ = writeln!(disassembly, "{name}:");
        for instruction in instructions {
            let _ = writeln!(disassembly, "{instruction}");
        }
    }

    let mut ir = String::from("vertex:\n");
    let mut msl = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         struct VertexOutput {\n    float4 position [[position]];\n",
    );
    if parameter.is_some() {
        msl.push_str("    float4 parameter [[user(locn0), center_perspective]];\n");
    }
    let _ = write!(
        msl,
        "}};\n\nvertex VertexOutput {VERTEX_ENTRY_POINT}(\n    uint vertex_id [[vertex_id]],\n    device const float4* positions [[buffer(0)]],\n    device const float4* colors [[buffer(1)]]) {{\n"
    );
    emit_values(&vertex, &mut ir, &mut msl);
    msl.push_str("    VertexOutput output;\n");
    emit_export(position, "pos0", "output.position", &mut ir, &mut msl);
    if let Some(parameter) = parameter {
        emit_export(parameter, "param0", "output.parameter", &mut ir, &mut msl);
    }
    msl.push_str(
        "    return output;\n}\n\nstruct PixelOutput {\n    float4 color [[color(0)]];\n};\n\n",
    );
    let mut params = Vec::new();
    if interpolates {
        params.push("VertexOutput input [[stage_in]]".to_string());
    }
    if has_sampling {
        params.push("texture2d<float, access::sample> tex0 [[texture(0)]]".to_string());
        params.push("sampler samp0 [[sampler(0)]]".to_string());
    }
    let param_str = params.join(",\n    ");
    let param_decl = if params.is_empty() {
        String::new()
    } else if params.len() == 1 {
        param_str
    } else {
        format!("\n    {param_str}")
    };
    let _ = write!(
        msl,
        "fragment PixelOutput {PIXEL_ENTRY_POINT}({param_decl}) {{\n    PixelOutput output;\n"
    );
    ir.push_str("pixel:\n");
    emit_values(&pixel, &mut ir, &mut msl);
    emit_export(color, "mrt0", "output.color", &mut ir, &mut msl);
    msl.push_str("    return output;\n}\n");
    Ok(CompiledShader {
        disassembly,
        ir,
        msl,
    })
}

fn lift(
    stage: Stage,
    decoded: &[DecodedInstruction],
    length: usize,
) -> Result<StageIr, CompileError> {
    let mut ir = StageIr::default();
    let mut registers = [None; 256];
    if stage == Stage::Vertex {
        let (positions, remaining) = registers.split_at_mut(4);
        for (slots, input) in [
            (positions, Float4Input::Position),
            (&mut remaining[..4], Float4Input::Color),
        ] {
            for (component, register) in slots.iter_mut().enumerate() {
                *register = Some(ir.push(None, Op::VertexInput(input, component as u8)));
            }
        }
    }

    let mut instructions = decoded.iter();
    let mut exporting = false;
    while let Some(&DecodedInstruction {
        instruction,
        offset,
        ..
    }) = instructions.next()
    {
        let error = |reason| CompileError::Verify { offset, reason };
        let read = |register: u8| {
            registers[usize::from(register)]
                .ok_or_else(|| error("source VGPR is uninitialized or reserved"))
        };
        match instruction {
            Instruction::MoveB32 { dst, src } if !exporting => {
                if stage == Stage::Pixel && dst < 2 {
                    return Err(error("pixel VGPRs 0 and 1 are reserved for interpolation"));
                }
                let value = match src {
                    VectorSource::Register(register) => read(register)?,
                    VectorSource::Immediate(bits) if f32::from_bits(bits).is_finite() => {
                        ir.push(Some(offset), Op::Constant(bits))
                    }
                    VectorSource::Immediate(_) => {
                        return Err(error("graphics move requires finite FP32 bits"));
                    }
                    VectorSource::Scalar(_) => {
                        return Err(error("graphics move from an SGPR is unsupported"));
                    }
                };
                registers[usize::from(dst)] = Some(value);
            }
            Instruction::VectorAlu {
                op: op @ (VectorAluOp::AddF32 | VectorAluOp::MulF32),
                dst,
                src0: VectorSource::Register(src0),
                src1,
            } if stage == Stage::Vertex && !exporting => {
                let operator = Binary::Vector(match op {
                    VectorAluOp::AddF32 => VectorBinary::AddF32,
                    _ => VectorBinary::MulF32,
                });
                let value = ir.push(Some(offset), Op::Binary(operator, read(src0)?, read(src1)?));
                registers[usize::from(dst)] = Some(value);
            }
            Instruction::Interpolate {
                phase: InterpolationPhase::First,
                dst,
                coordinate: 0,
                attribute: 0,
                component,
            } if stage == Stage::Pixel && dst >= 2 && !exporting => {
                // Accept only an adjacent, matching pair. The intermediate
                // interpolation result is never observable independently.
                let expected = Instruction::Interpolate {
                    phase: InterpolationPhase::Second,
                    dst,
                    coordinate: 1,
                    attribute: 0,
                    component,
                };
                let second = instructions
                    .next()
                    .ok_or_else(|| error("missing matching second interpolation phase"))?;
                if second.instruction != expected {
                    return Err(CompileError::Verify {
                        offset: second.offset,
                        reason: "expected matching second interpolation phase",
                    });
                }
                registers[usize::from(dst)] =
                    Some(ir.push(Some(offset), Op::Interpolate(component)));
            }
            Instruction::ImageSampleLz {
                dst,
                coord,
                resource,
                sampler,
                dmask: 0xf,
            } if stage == Stage::Pixel && dst >= 2 && !exporting => {
                // In synthetic binding ABI:
                // resource s[0:7] maps to texture(0)
                // sampler s[8:11] maps to sampler(0)
                if (resource, sampler) != (0, 8) {
                    return Err(error(
                        "sampling outside synthetic resource binding ABI s[0:7] and s[8:11]",
                    ));
                }
                let coord_end = coord
                    .checked_add(1)
                    .ok_or_else(|| error("image sample coordinate pair exceeds v255"))?;
                let dst_end = dst
                    .checked_add(3)
                    .ok_or_else(|| error("image sample result quartet exceeds v255"))?;
                let u = read(coord)?;
                let v = read(coord_end)?;
                for (component, register) in
                    (0..4).zip(&mut registers[usize::from(dst)..=usize::from(dst_end)])
                {
                    *register = Some(ir.push(
                        Some(offset),
                        Op::SampleLodZero {
                            u,
                            v,
                            component,
                            texture_index: 0,
                            sampler_index: 0,
                        },
                    ));
                }
            }
            Instruction::Export {
                target,
                sources: [a, b, c, d],
            } => {
                let slot = match (stage, target) {
                    (Stage::Vertex, ExportTarget::Position0) => &mut ir.position,
                    (Stage::Vertex, ExportTarget::Parameter0) => &mut ir.parameter,
                    (Stage::Pixel, ExportTarget::RenderTarget0) => &mut ir.color,
                    _ => return Err(error("export target is invalid for this stage")),
                };
                if slot.is_some() {
                    return Err(error("export target is written more than once"));
                }
                *slot = Some(Export {
                    offset,
                    values: [read(a)?, read(b)?, read(c)?, read(d)?],
                });
                exporting = true;
            }
            Instruction::EndProgram => {
                if let Some(next) = instructions.next() {
                    return Err(CompileError::Verify {
                        offset: next.offset,
                        reason: "instruction follows program termination",
                    });
                }
                return Ok(ir);
            }
            _ => {
                return Err(error(
                    "instruction is unsupported at this point in the graphics stage",
                ));
            }
        }
    }
    Err(CompileError::Verify {
        offset: length,
        reason: "missing program termination",
    })
}

fn emit_values(stage: &StageIr, ir: &mut String, msl: &mut String) {
    if stage
        .values
        .iter()
        .any(|(_, op)| matches!(op, Op::Binary(..)))
    {
        msl.push_str("    #pragma clang fp contract(off)\n    #pragma clang fp reassociate(off)\n");
    }

    for (index, &(offset, op)) in stage.values.iter().enumerate() {
        let value = Value(index);
        let (operation, expression) = match op {
            Op::VertexInput(input, component) => {
                let buffer = match input {
                    Float4Input::Position => "positions",
                    Float4Input::Color => "colors",
                };
                (
                    format!("load.vertex.f32 {buffer}[{component}]"),
                    format!("{buffer}[vertex_id][{component}]"),
                )
            }
            Op::Constant(bits) => (
                format!("const.f32 0x{bits:08x}"),
                format!("as_type<float>(0x{bits:08x}u)"),
            ),
            Op::Binary(operator, left, right) => {
                let (mnemonic, symbol) = operator.operator();
                (
                    format!("{mnemonic} {left}, {right}"),
                    format!("{left} {symbol} {right}"),
                )
            }
            Op::Interpolate(component) => (
                format!("interpolate.center_perspective.f32 attr0[{component}]"),
                format!("input.parameter[{component}]"),
            ),
            Op::SampleLodZero {
                u,
                v,
                component,
                texture_index,
                sampler_index,
            } => {
                let channel = ["r", "g", "b", "a"][usize::from(component)];
                (
                    format!(
                        "sample_lz.2d.f32 tex{texture_index}, samp{sampler_index}, ({u}, {v})[{channel}]"
                    ),
                    format!(
                        "tex{texture_index}.sample(samp{sampler_index}, float2({u}, {v}), level(0.0f)).{channel}"
                    ),
                )
            }
        };
        if let Some(offset) = offset {
            let _ = write!(ir, "{offset:04x}: ");
        }
        let _ = writeln!(ir, "{value} = {operation}");
        let _ = writeln!(msl, "    float {value} = {expression};");
    }
}

fn emit_export(export: &Export, target: &str, field: &str, ir: &mut String, msl: &mut String) {
    let [a, b, c, d] = export.values;
    let offset = export.offset;
    let _ = writeln!(
        ir,
        "{offset:04x}: export.f32x4 {target}, {a}, {b}, {c}, {d}"
    );
    let _ = writeln!(msl, "    {field} = float4({a}, {b}, {c}, {d});");
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_OFFSET: usize = 8;

    fn pixel_sample(dst: u8, coord: u8, resource: u8) -> Result<StageIr, CompileError> {
        let init = |offset, dst, value: f32| DecodedInstruction {
            offset,
            byte_len: 4,
            instruction: Instruction::MoveB32 {
                dst,
                src: VectorSource::Immediate(value.to_bits()),
            },
        };
        let instructions = [
            init(0, 254, 0.25),
            init(4, 255, 0.75),
            DecodedInstruction {
                offset: SAMPLE_OFFSET,
                byte_len: 8,
                instruction: Instruction::ImageSampleLz {
                    dst,
                    coord,
                    resource,
                    sampler: 8,
                    dmask: 0xf,
                },
            },
            DecodedInstruction {
                offset: 16,
                byte_len: 8,
                instruction: Instruction::Export {
                    target: ExportTarget::RenderTarget0,
                    sources: [252, 253, 254, 255],
                },
            },
            DecodedInstruction {
                offset: 24,
                byte_len: 4,
                instruction: Instruction::EndProgram,
            },
        ];
        lift(Stage::Pixel, &instructions, 28)
    }

    #[test]
    fn samples_with_upper_register_spans() {
        let ir = pixel_sample(252, 254, 0).unwrap();
        assert_eq!(ir.color.unwrap().values.map(|value| value.0), [2, 3, 4, 5]);
        for (component, (offset, op)) in (0..4).zip(&ir.values[2..]) {
            assert_eq!(*offset, Some(SAMPLE_OFFSET));
            assert!(matches!(
                op,
                Op::SampleLodZero { u: Value(0), v: Value(1), component: actual, .. }
                    if *actual == component
            ));
        }
    }

    #[test]
    fn rejects_invalid_sample_spans_before_reading_operands() {
        for (dst, coord, resource, expected) in [
            (252, 255, 0, "image sample coordinate pair exceeds v255"),
            (253, 254, 0, "image sample result quartet exceeds v255"),
            (255, 2, 0, "image sample result quartet exceeds v255"),
            (252, 2, 0, "source VGPR is uninitialized or reserved"),
            (
                253,
                255,
                4,
                "sampling outside synthetic resource binding ABI s[0:7] and s[8:11]",
            ),
        ] {
            assert!(
                matches!(
                    pixel_sample(dst, coord, resource),
                    Err(CompileError::Verify { offset: SAMPLE_OFFSET, reason })
                        if reason == expected
                ),
                "expected a failure at the sample instruction for v{dst}, v{coord}, s{resource}"
            );
        }
    }
}
