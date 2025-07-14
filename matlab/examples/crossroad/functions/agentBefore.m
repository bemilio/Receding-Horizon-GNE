function j = agentBefore(i, crossing_dir, conflicts_with)
    % Find agent that i has to wait before passing
    conflicts_with_i = conflicts_with(crossing_dir(i));
    all_conflicting_agents = find(ismember(crossing_dir, conflicts_with_i) );
    all_conflicting_agents(all_conflicting_agents >= i) = [];
    j = max(all_conflicting_agents);
end
