use crate::algorithm::{
    MeshAlgorithm, MeshAlgorithmCapabilities, MeshAlgorithmDescriptor, MeshSink, MeshingContext,
    MeshingStatistics,
};
use crate::error::{MeshError, MeshResult};

pub static ADVANCING_FRONT: AdvancingFront = AdvancingFront;
pub static ADVANCING_FRONT_DESCRIPTOR: MeshAlgorithmDescriptor = MeshAlgorithmDescriptor {
    id: "advancing_front",
    label: "Advancing Front",
    dimensions: &[2, 3],
    capabilities: MeshAlgorithmCapabilities {
        refinement: true,
        boundary_layers: false,
    },
};

#[derive(Debug, Clone, Copy, Default)]
pub struct AdvancingFront;

impl MeshAlgorithm for AdvancingFront {
    fn descriptor(&self) -> &'static MeshAlgorithmDescriptor {
        &ADVANCING_FRONT_DESCRIPTOR
    }

    fn generate(
        &self,
        context: &MeshingContext<'_>,
        sink: &mut dyn MeshSink,
    ) -> MeshResult<MeshingStatistics> {
        match context.domains.iter().next().map(|domain| domain.dimension) {
            Some(2) => crate::advancing_front_2d::generate(context, sink),
            Some(3) => crate::advancing_front_3d::generate(context, sink),
            Some(dimension) => Err(MeshError::UnsupportedDimension {
                domain: context
                    .domains
                    .iter()
                    .next()
                    .map(|domain| domain.name.clone())
                    .unwrap_or_default(),
                dimension,
            }),
            None => Err(MeshError::InvalidInput(
                "meshing requires at least one domain".into(),
            )),
        }
    }
}
