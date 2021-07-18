use petgraph::dot::Dot;
use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
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
    output: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Node {
    name: PathBuf,
    ty: String,
}

impl Node {
    fn from_operation(op: &Operation) -> Self {
        let name = PathBuf::from(op.name().expect("Op name not valid unicode"));
        let ty = op.op_type().expect("Op type not valid unicode");
        Self { name, ty }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Edge {
    dim: Vec<Option<usize>>,
    input_index: Option<usize>,
    output_index: Option<usize>,
}

fn add_op_to_graph(
    input: (&Operation, Option<usize>),
    output: (&Operation, Option<usize>),
    graph: &mut Graph<Node, Edge>,
    nodes: &mut HashMap<Node, NodeIndex>,
) {
    let in_node = Node::from_operation(&input.0);
    let in_idx = *nodes
        .entry(in_node.clone())
        .or_insert_with(|| graph.add_node(in_node.clone()));
    let out_node = Node::from_operation(&output.0);
    let out_idx = *nodes
        .entry(out_node.clone())
        .or_insert_with(|| graph.add_node(out_node.clone()));
    if graph.find_edge(in_idx, out_idx).is_none() {
        let edge = Edge {
            dim: vec![],
            input_index: input.1,
            output_index: output.1,
        };
        graph.add_edge(in_idx, out_idx, edge);
    }
}

fn generate_graph(nn_graph: &TfGraph) -> Graph<Node, Edge> {
    let mut graph = Graph::<Node, Edge>::new();
    let mut nodes = HashMap::new();
    for op in nn_graph.operation_iter() {
        for i in 0..op.num_inputs() {
            let (input, idx) = op.input(i);
            add_op_to_graph((&op, Some(i)), (&input, Some(idx)), &mut graph, &mut nodes);
        }
        for i in 0..op.num_outputs() {
            for (output, idx) in op.output_consumers(i).iter() {
                add_op_to_graph(
                    (&output, Some(*idx)),
                    (&op, Some(i)),
                    &mut graph,
                    &mut nodes,
                );
            }
        }
        for output in op.control_outputs().iter() {
            add_op_to_graph((&output, None), (&op, None), &mut graph, &mut nodes);
        }
        for input in op.control_inputs().iter() {
            add_op_to_graph((&op, None), (&input, None), &mut graph, &mut nodes);
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
    let dot = Dot::new(&graph);

    if let Some(o) = config.output {
        let mut file = fs::File::create(o)?;
        file.write_all(format!("{:?}", dot).as_bytes())?;
    } else {
        println!("{:?}", dot);
    }

    Ok(())
}
