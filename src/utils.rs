pub fn transpose<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let Some(first_row) = matrix.first() else {
        return vec![];
    };
    let row_count = matrix.len();
    let column_count = first_row.len();

    let mut transposed: Vec<Vec<T>> = Vec::with_capacity(column_count);
    for column_index in 0..column_count {
        let mut new_row: Vec<T> = Vec::with_capacity(row_count);
        for row in matrix {
            new_row.push(row[column_index].clone());
        }
        transposed.push(new_row);
    }
    transposed
}