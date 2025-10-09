{
  description = "Dev environment for Tosa Nikki-based RAG";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url  = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ self, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # Expose outputs for all major platforms
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      
      perSystem = { pkgs, inputs', lib, system, ... }: {
        # Override the provider's default devShell while reusing it as a base
        devShells.default = lib.mkForce (pkgs.mkShell {
          packages = with pkgs; [
            (python312.withPackages (p: with p; [
              neo4j
              pydantic
              openai
              sentence-transformers
            ]))
          ];
          shellHook = ''
            echo "Tosa Nikki RAG dev"

            export OPENAI_API_KEY=$(cat params/api-keys/openai)
            export NEO4J_URI=$(cat params/neo4j/uri)
            export NEO4J_USERNAME=$(cat params/neo4j/username)
            export NEO4J_PASSWORD=$(cat params/neo4j/password)
          '';
        });
      };
    };
}
