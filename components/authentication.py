from ldap3 import Server, Connection


class GetW2K:
    def __init__(self) -> None:
        self.server = Server("ldap://sq.com.sg:3268")

    def authenticate(self, user_name, pwd):
        user_email = user_name + "@singaporeair.com.sg"
        connection = Connection(self.server, user=user_email, password=pwd)
        bind_response = connection.bind()  # Returns True or False

        if bind_response:
            connection.search(
                "dc=com,dc=sg",
                "(&(objectClass=user)(sAMAccountName=" + user_name + "))",
                attributes=["title", "department"],
            )
            entry = connection.entries[0]
            return entry.department.value + "|" + entry.title.value
        else:
            return "Not valid"
