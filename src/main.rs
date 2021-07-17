use petgraph::dot::Dot;
use petgraph::graph::Graph;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use structopt::StructOpt;
use tensorflow::{Graph as TfGraph, ImportGraphDefOptions, Operation};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, StructOpt)]
pub struct Config {
    /// Input neural network to render
    #[structopt(short, long)]
    input: PathBuf,
    /// Save rendered output here
    #[structopt(short, long)]
    output: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Node {
    name: String,
    ty: String,
}

impl Node {
    fn from_operation(op: &Operation) -> Self {
        let name = op.name().expect("Op name not valid unicode");
        let ty = op.op_type().expect("Op type not valid unicode");
        Self { name, ty }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Edge {
    dim: Vec<Option<usize>>,
    input_index: usize,
    output_index: usize,
}

fn generate_graph(nn_graph: &TfGraph) -> Graph<Node, Edge> {
    let mut graph = Graph::<Node, Edge>::new();
    let mut nodes = HashMap::new();
    for op in nn_graph.operation_iter() {
        let node = Node::from_operation(&op);
        let node_idx = nodes
            .entry(node.clone())
            .or_insert_with(|| graph.add_node(node.clone()))
            .clone();
        for i in 0..op.num_inputs() {
            let (input, idx) = op.input(i);
            let in_node = Node::from_operation(&input);
            let in_idx = nodes
                .entry(in_node.clone())
                .or_insert_with(|| graph.add_node(in_node.clone()));
            let edge = Edge {
                dim: vec![],
                input_index: i,
                output_index: idx,
            };
            graph.add_edge(*in_idx, node_idx, edge);
        }
        for i in 0..op.num_outputs() {
            let (output, idx) = op.input(i);
            let out_node = Node::from_operation(&output);
            if nodes.contains_key(&out_node) {
                continue;
            }
            let out_idx = nodes
                .entry(out_node.clone())
                .or_insert_with(|| graph.add_node(out_node.clone()));

            let edge = Edge {
                dim: vec![],
                input_index: idx,
                output_index: i,
            };
            graph.add_edge(*out_idx, node_idx, edge);
        }
    }
    graph
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_args();

    let input = fs::read(&config.input)?;

    let mut graph = TfGraph::new();
    graph.import_graph_def(&input, &ImportGraphDefOptions::new())?;

    let graph = generate_graph(&graph);

    let dot = Dot::new(graph);

    println!("{:?}", dot);

    Ok(())
}
