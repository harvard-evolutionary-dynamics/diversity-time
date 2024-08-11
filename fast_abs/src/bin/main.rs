use fast_abs::moran::{self, BELLS_NUMBER};

extern crate nalgebra as na;
extern crate ndarray as nd;

use nd::ShapeBuilder;

fn main() {
  let mut birth_event: moran::BirthEventMap =
    nd::Array::<usize, _>::zeros((moran::BELLS_NUMBER, moran::N, moran::N).f());
  moran::load_birth_events(&mut birth_event);
  println!("Loaded birth events");

  let mut graphs: Vec<moran::AdjacencyMatrix> = Vec::new();
  graphs.resize(moran::NUM_GRAPHS, moran::AdjacencyMatrix::zeros(moran::N, moran::N));
  moran::load_graphs(&mut graphs);
  println!("Loaded graphs");

  let mut birth_death_matrices: Vec<moran::MoranMatrix> = Vec::new();
  birth_death_matrices.resize(moran::NUM_GRAPHS, moran::MoranMatrix::zeros(moran::BELLS_NUMBER, moran::BELLS_NUMBER));
  moran::load_moran_matrices(&graphs, &birth_event, &mut birth_death_matrices, true);
  println!("Loaded moran matrices");

  let mut birth_death_results: Vec<f32> = Vec::new();
  birth_death_results.resize(moran::NUM_GRAPHS, 0.);
  moran::solve_moran_matrices(&birth_death_matrices, &mut birth_death_results);
  println!("Solved moran matrices");

  for result in birth_death_results {
    println!("Result: {result}");
  }
}

