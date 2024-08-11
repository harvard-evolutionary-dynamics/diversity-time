pub mod moran {
  extern crate nalgebra as na;
  extern crate ndarray as nd;

  use std::fs::File;
  use std::io::BufReader;
  use std::io::BufRead;
  use csv::ReaderBuilder;

  /// Number of nodes in the graph.
  pub const N: usize = 7; // 8;
  /// OEIS A000110.
  pub const BELLS_NUMBER: usize = 877; // 4140;
  /// OEIS A001349.
  pub const NUM_GRAPHS: usize = 853; // 11117;

  pub type AdjacencyMatrix = na::DMatrix<f32>; // , N, N>;
  pub type MoranMatrix = na::DMatrix<f32>; // , BELLS_NUMBER, BELLS_NUMBER>;
  pub type BirthEventMap = nd::Array::<usize, nd::Dim<[usize; 3]>>;

  pub fn solve_moran_matrices(
    matrices: &Vec<MoranMatrix>,
    results: &mut Vec<f32>,
  ) {

    for graph_idx in 0..NUM_GRAPHS {
      println!("{graph_idx} ");
      let b: na::DVector<f32> =
        na::DVector::from_fn(BELLS_NUMBER, |i, _| if i == 0 { 0. } else { 1. });
      let mut mat_a_inv = MoranMatrix::zeros(BELLS_NUMBER, BELLS_NUMBER);
      let mat_a = &matrices[graph_idx];
      na::linalg::try_invert_to(mat_a.clone(), &mut mat_a_inv);
      let x: na::DVector<f32> = mat_a_inv * b;
      results[graph_idx] = x[BELLS_NUMBER-1];
    }
  }

  /// Input matrices should be zeroed out.
  pub fn load_moran_matrices(
    graphs: &Vec<AdjacencyMatrix>,
    birth_event: &BirthEventMap,
    matrices: &mut Vec<MoranMatrix>,
    birth_death: bool,
  ) {
    for graph_idx in 0..NUM_GRAPHS {
      let degrees = graphs[graph_idx].row_sum();
      let mat_a = &mut matrices[graph_idx];
      for s_idx in 0..BELLS_NUMBER {
        mat_a[(s_idx, s_idx)] = 1.;

        // This is the absorbing state.
        if s_idx == 0 { continue }

        for u in 0..N {
          for v in 0..N {
            if graphs[graph_idx][(u, v)] == 0. { continue }
            let t_idx = birth_event[[s_idx, u, v]]; // birth_event[s_idx][u][v];
            let pivotal_node = if birth_death { u } else { v };
            let normalization = 1. / degrees[pivotal_node];
            mat_a[(s_idx, t_idx)] += (-1./(N as f32)) * normalization;
          }
        }
      }
    }
  }

  pub fn load_birth_events(birth_event: &mut BirthEventMap) {
    let file = File::open(format!("../data/events-{N}.csv")).unwrap();
  
    let mut csv_reader = ReaderBuilder::new()
      .has_headers(true)
      .from_reader(file);
  
    for result in csv_reader.records() {
      let record = result.unwrap();
      let s_idx: usize = record[0].parse().unwrap();
      let u: usize = record[1].parse().unwrap();
      let v: usize = record[2].parse().unwrap();
      let t_idx: usize = record[3].parse().unwrap();
      birth_event[[s_idx, u, v]] = t_idx;
    }
  }

  pub fn load_graphs(graphs: &mut Vec<AdjacencyMatrix>) {
    let file = File::open(format!("../data/connected-n{N}.g6")).unwrap();
    let reader = BufReader::new(file);
    for (idx, line) in reader.lines().enumerate() {
      let graph_repr = line.unwrap();
      let adjacency_matrix_rows = graph6::string_to_adjacency_matrix(&graph_repr);
      let graph = AdjacencyMatrix::from_row_slice(N, N, adjacency_matrix_rows.0.as_slice());
      graphs[idx] = graph;
    }
  }
}
