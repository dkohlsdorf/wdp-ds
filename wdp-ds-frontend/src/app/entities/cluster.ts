export interface Cluster {
    id:  bigint;
    encoding: bigint;
    filename: string;
    start: number;
    stop: number;
    algorithm: string;
    cluster_id: bigint;
    created_at: string;
}